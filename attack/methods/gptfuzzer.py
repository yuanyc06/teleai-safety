"""
GPTFUZZER Attack Method
============================================
This class implements a jailbreak method described in the paper below.
This part of code is based on the code from the paper.

Paper title: GPTFUZZER : Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts
arXiv link: https://arxiv.org/pdf/2309.10253.pdf
Source repository: https://github.com/sherdencooper/GPTFuzz.git
"""

import os
import sys
import json
import time
import copy
import random
import requests
from typing import Optional, List, Dict, Any

# Add current working directory to path for relative imports
sys.path.append(os.getcwd())

# Third-party imports
from tqdm import tqdm
from loguru import logger
from dataclasses import dataclass, field, fields

# Local imports
from dataset import AttackDataset
from dataset.example import Example
from utils import BaseAttackManager, ConfigManager, parse_arguments, Timer
from models import load_model
from logger import setup_logger
from initialization.seed_template import SeedTemplate
from selection.mcts_selection import MctsSelection
from mutation.crossover import GPTFuzzerCrossOver
from mutation import Expand, GenerateSimilar, Shorten, Rephrase

@dataclass
class GPTFuzzerConfig:
    """Configuration class for GPTFuzzer attack parameters."""
    attack_data_path: str
    attack_model_path: str
    target_model_path: str
    attack_model_name: str
    target_model_name: str
    attack_model_type: str = 'local'  # openai, azure, local
    target_model_type: str = 'local'  # openai, azure, local
    attack_model_generation_config: dict = field(default_factory=lambda: {"max_new_tokens": 512, "do_sample": True, "temperature": 0.8})
    target_model_generation_config: dict = field(default_factory=lambda: {"max_new_tokens": 512, "do_sample": True, "temperature": 0.8})
    eval_model_path: str = "hubert233/GPTFuzz"
    seeds_num: int = 76
    energy: int = 5
    max_query: int = 100000
    max_jailbreak: int = 100000
    max_reject: int = 100000
    max_iteration: int = 100
    template_file: Optional[str] = None
    device: str = 'cuda'
    
    # API keys for different services
    api_key: str = ''  # OpenAI API key
    base_url: str = ''  # OpenAI base URL
    azure_key: str = ''  # Azure API key
    azure_url: str = ''  # Azure endpoint URL
    grok_key: str = ''  # Grok API key  
    grok_url: str = ''  # Grok endpoint URL
    
    res_save_path: Optional[str] = None
    log_path: Optional[str] = None


@dataclass
class AttackData:
    attack_results: Optional[AttackDataset] = AttackDataset([])
    query_to_idx: Optional[dict] = field(default_factory=dict)
    questions: Optional[AttackDataset] = AttackDataset([])
    prompt_nodes: Optional[AttackDataset] = AttackDataset([])
    initial_prompts_nodes: Optional[AttackDataset] = AttackDataset([])
    query: str = None
    jailbreak_prompt: str = None
    reference_responses: List[str] = field(default_factory=list)
    target_responses: List[str] = field(default_factory=list)
    eval_results: list = field(default_factory=list)
    attack_attrs: dict = field(default_factory=lambda: {'Mutation': None, 'query_class': None})
    parents: list = field(default_factory=list)
    children: list = field(default_factory=list)
    index: int = None
    visited_num: int = 0
    level: int = 0
    current_query: int = 0
    current_jailbreak: int = 0
    current_reject: int = 0
    total_query: int = 0
    total_jailbreak: int = 0
    total_reject: int = 0
    current_iteration: int = 0
    _num_query: int = 0
    _num_jailbreak: int = 0
    _num_reject: int = 0

    def copy(self):
        return copy.deepcopy(self)

    @property
    def num_query(self):
        return len(self.target_responses)

    @property
    def num_jailbreak(self):
        return sum(i for i in self.eval_results)

    @property
    def num_reject(self):
        return len(self.eval_results) - sum(i for i in self.eval_results)
    
    def copy_for_eval(self):
        """
        Lightweight copy used during evaluation to avoid deepcopy overhead.
        """
        return AttackData(
            query=self.query,
            jailbreak_prompt=self.jailbreak_prompt,
            attack_attrs=self.attack_attrs.copy(),
            index=self.index,
            level=self.level
        )


class GPTFuzzerInit(object):
    """
    Initializes the attack data structure and seeds.
    """
    def __init__(self, config: GPTFuzzerConfig, jailbreak_datasets: Optional[AttackDataset] = None):
        self.config = config
        
        self.data = AttackData(questions=jailbreak_datasets)
        self.data.questions_length = len(self.data.questions) if jailbreak_datasets else 0

        # Initialize seeds and prompts
        self.data.initial_prompt_seed = SeedTemplate().new_seeds(
            seeds_num=config.seeds_num, prompt_usage='attack',
            method_list=['Gptfuzzer'], template_file=config.template_file
        )
        
        self.data.prompt_nodes = AttackDataset(
            [AttackData(jailbreak_prompt=prompt) for prompt in self.data.initial_prompt_seed]
        )
        for i, instance in enumerate(self.data.prompt_nodes):
            instance.index = i
            instance.visited_num = 0
            instance.level = 0
        for i, instance in enumerate(self.data.questions):
            instance.index = i
        self.data.initial_prompts_nodes = AttackDataset([instance for instance in self.data.prompt_nodes])


class GPTFuzzerMutation(object):
    """
    Handles mutations for generating new prompts during the attack.
    """
    def __init__(self, attack_model, initial_prompts_nodes):

        self.mutations = [
            GPTFuzzerCrossOver(attack_model, seed_pool=initial_prompts_nodes),
            Expand(attack_model),
            GenerateSimilar(attack_model),
            Shorten(attack_model),
            Rephrase(attack_model)
        ]
        
        self.mutator = None
        
    def mutate(self, instance):
        try:
            self.mutator = random.choice(self.mutations)
            result = self.mutator._get_mutated_instance(instance)
            
            # Validate mutation result
            if not result or len(result) == 0:
                logger.warning(f"Mutation failed: empty result from {type(self.mutator).__name__}")
                return [instance]  # Return original instance as fallback
                
            mutated_instance = result[0]
            if not hasattr(mutated_instance, 'jailbreak_prompt') or not mutated_instance.jailbreak_prompt:
                logger.warning(f"Mutation failed: empty jailbreak_prompt from {type(self.mutator).__name__}")
                return [instance]  # Return original instance as fallback
                
            logger.debug(f"Mutation successful: {type(self.mutator).__name__}")
            return result
            
        except Exception as e:
            logger.error(f"Exception in mutation: {e}")
            return [instance]  # Return original instance as fallback


class GPTFuzzerEvaluator(object):
    """
    Evaluates the results of the generated prompts.
    """
    def __init__(self, eval_model=None, eval_model_path=None):
        assert eval_model is not None or eval_model_path is not None, "At least one of eval_model or eval_model_path must be provided"
        # Import locally to avoid circular imports
        try:
            from evaluation import ClassficationScorer
            self.evaluator = ClassficationScorer(eval_model)
            self.evaluator.set_model(eval_model_path)
        except ImportError:
            logger.warning("ClassficationScorer not available, using dummy evaluator")
            self.evaluator = lambda x: None


class APIModel:
    """Generic API model for different providers."""
    
    def __init__(self, model_name: str, base_url: str, api_key: str, system_prompt: str = "You are a helpful assistant."):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.system_prompt = system_prompt

    def chat(self, prompt: str) -> str:
        """Single chat method for individual prompts."""
        url = self.base_url
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        data = {
            "messages": messages,
            "max_completion_tokens": 2048,
            "temperature": 1,
            "top_p": 1,
            "model": self.model_name
        }
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.warning(f"Unexpected API response format: {result}")
                return str(result)
        except requests.exceptions.Timeout:
            logger.warning("API request timed out")
            return "[API Timeout]"
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed: {e}")
            return f"[API Error] {e}"
        except Exception as e:
            logger.warning(f"Unexpected error in API call: {e}")
            return f"[API Error] {e}"

    def batch_chat(self, prompts: List[str]) -> List[str]:
        """Batch chat method for multiple prompts."""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        responses = []
        logger.info(f"Processing {len(prompts)} prompts with API...")
        for i, prompt in enumerate(tqdm(prompts, desc="API calls")):
            try:
                response = self.chat(prompt)
                responses.append(response)
                if (i + 1) % 10 == 0:
                    logger.debug(f"Completed {i + 1}/{len(prompts)} API calls")
            except Exception as e:
                logger.warning(f"Error processing prompt {i+1}: {e}")
                responses.append(f"[Error] {e}")
        logger.info(f"Completed all {len(prompts)} API calls")
        return responses


class GPTFuzzerManager(BaseAttackManager):
    """
    Main class for managing and executing GPTFuzzer attacks.
    
    This attack method uses evolutionary fuzzing techniques to create adversarial prompts
    that can potentially bypass safety mechanisms in language models.
    """

    def __init__(
        self,
        attack_data_path: str = "thu-coai/AISafetyLab_Datasets/harmbench_standard",
        attack_model_path: str = "/gemini/code/Qwen2.5-7B-Instruct",
        target_model_path: str = "/gemini/code/Qwen2.5-7B-Instruct",
        attack_model_name: str = "qwen2.5",
        target_model_name: str = "qwen2.5",
        attack_model_type: str = 'local',
        target_model_type: str = 'local',
        attack_model_generation_config: dict = None,
        target_model_generation_config: dict = None,
        eval_model_path: str = "hubert233/GPTFuzz",
        seeds_num: int = 76,
        energy: int = 5,
        max_query: int = 100000,
        max_jailbreak: int = 100000,
        max_reject: int = 100000,
        max_iteration: int = 100,
        template_file: str = "aisafetylab/attack/initialization/init_templates.json",
        device: str = "cuda:0",
        res_save_path: str = "./results/gptfuzzer_results.jsonl",
        log_path: str = '../logs/',
        subset_slice: Optional[slice] = None,
        api_key: str = "",
        base_url: str = "",
        azure_key: str = "",
        azure_url: str = "",
        grok_key: str = "",
        grok_url: str = "",
        *args,
        **kwargs
    ):
        """Initialize the GPTFuzzer attack manager."""
        super().__init__(res_save_path=res_save_path)

        # Set default generation configs if not provided
        if attack_model_generation_config is None:
            attack_model_generation_config = {"max_new_tokens": 512, "do_sample": True, "temperature": 0.8}
        if target_model_generation_config is None:
            target_model_generation_config = {"max_new_tokens": 512, "do_sample": True, "temperature": 0.8}

        # Create configuration object
        self.config = GPTFuzzerConfig(
            attack_data_path=attack_data_path,
            attack_model_path=attack_model_path,
            target_model_path=target_model_path,
            attack_model_name=attack_model_name,
            target_model_name=target_model_name,
            attack_model_type=attack_model_type,
            target_model_type=target_model_type,
            attack_model_generation_config=attack_model_generation_config,
            target_model_generation_config=target_model_generation_config,
            eval_model_path=eval_model_path,
            seeds_num=seeds_num,
            energy=energy,
            max_query=max_query,
            max_jailbreak=max_jailbreak,
            max_reject=max_reject,
            max_iteration=max_iteration,
            template_file=template_file,
            device=device,
            api_key=api_key,
            base_url=base_url,
            azure_key=azure_key,
            azure_url=azure_url,
            grok_key=grok_key,
            grok_url=grok_url,
            res_save_path=res_save_path,
            log_path=log_path
        )

        # Initialize components
        self._initialize_components(subset_slice)
        
        # Setup logging
        if self.config.log_path:
            setup_logger(log_file_path=self.config.log_path)
        
        logger.info(f"GPTFuzzer Manager initialized with {len(self.jailbreak_datasets)} attack examples")
        logger.info(f"Attack model type: {self.config.attack_model_type}")
        logger.info(f"Target model type: {self.config.target_model_type}")
        logger.info(f"Attack model: {self.config.attack_model_name}")
        logger.info(f"Target model: {self.config.target_model_name}")

    def _initialize_components(self, subset_slice: Optional[slice] = None) -> None:
        """Initialize all components needed for the attack."""
        logger.info(f"Attack configuration: {self.config}")

        # Load models
        self.attack_model = load_model(
            model_type=self.config.attack_model_type,
            model_name=self.config.attack_model_name,
            model_path=self.config.attack_model_path,
            config=self.config
        )

        self.target_model = load_model(
            model_type=self.config.target_model_type,
            model_name=self.config.target_model_name,
            model_path=self.config.target_model_path,
            config=self.config
        )

        logger.info(f"Target model type: {type(self.target_model).__name__}")
        logger.info(f"Target model name: {self.config.target_model_name}")

        # Load dataset
        self.jailbreak_datasets = AttackDataset(self.config.attack_data_path, subset_slice=subset_slice)

        # Initialize components
        self.gptfuzzer_init = GPTFuzzerInit(self.config, self.jailbreak_datasets)
        self.data = self.gptfuzzer_init.data

        self.select_policy = MctsSelection(self.data.prompt_nodes, self.data.initial_prompts_nodes, self.data.questions)

        self.mutator = GPTFuzzerMutation(
            self.attack_model,
            self.data.initial_prompts_nodes
        )

    def mutate(self, query, target="Sure, here's"):
        # Wrap the query into an Example instance for mutation
        example_instance = AttackData(query=query, jailbreak_prompt=target)

        # Apply a mutation operation using the mutator
        mutated_instance = self.mutator.mutate(example_instance)[0]

        # Check if mutation was successful
        if not mutated_instance.jailbreak_prompt or mutated_instance.jailbreak_prompt.strip() == "":
            logger.warning(f"Mutation failed, jailbreak_prompt is empty. Using fallback.")
            mutated_instance.jailbreak_prompt = target  # Use original target as fallback

        # Generate the final query by applying the jailbreak prompt
        if '{query}' in mutated_instance.jailbreak_prompt:
            final_query = mutated_instance.jailbreak_prompt.format(query=mutated_instance.query)
        else:
            final_query = mutated_instance.jailbreak_prompt + " " + mutated_instance.query
        
        logger.debug(f"Original query: {query}")
        logger.debug(f"Jailbreak prompt: {mutated_instance.jailbreak_prompt}")
        logger.debug(f"Final query: {final_query}")
        
        return final_query

    def attack(self):
        """
        Main loop for the fuzzing process, repeatedly selecting, mutating, evaluating, and updating.
        """
        logger.info("Fuzzing started!")
        example_idx_to_results = {}
        try:
            while not self.is_stop():
                seed_instance = self.select_policy.select()[0]
                logger.debug('begin to mutate')
                mutated_results = self.single_attack(seed_instance)
                for instance in mutated_results:
                    instance.parents = [seed_instance]
                    instance.children = []
                    seed_instance.children.append(instance)
                    instance.index = len(self.data.prompt_nodes)
                    self.data.prompt_nodes.data.append(instance)
                logger.debug(f'mutated results length: {len(mutated_results)}')
                for mutator_instance in tqdm(mutated_results, desc="Processing mutated instances"):
                    self.temp_results = AttackDataset([])
                    logger.info('Begin to get model responses')
                    input_seeds = []
                    for query_instance in tqdm(self.data.questions, desc="Preparing input seeds"):
                        if '{query}' in mutator_instance.jailbreak_prompt:
                            input_seed = mutator_instance.jailbreak_prompt.replace('{query}', query_instance.query)
                        else:
                            input_seed = mutator_instance.jailbreak_prompt + query_instance.query
                        input_seeds.append(input_seed)
                    
                    logger.info(f"Sending {len(input_seeds)} queries to target model...")
                    all_responses = self.target_model.batch_chat(input_seeds)
                    logger.info("Received all responses from target model")
                            
                    for idx, query_instance in enumerate(tqdm(self.data.questions, desc="Processing responses")):
                        # logger.debug(f'mutator_instance: {mutator_instance}')
                        # t = Timer.start()
                        temp_instance = mutator_instance.copy_for_eval()
                        # print(f'Copy costs {t.end()} seconds')
                        temp_instance.target_responses = []
                        temp_instance.eval_results = []
                        temp_instance.query = query_instance.query

                        if temp_instance.query not in self.data.query_to_idx:
                            self.data.query_to_idx[temp_instance.query] = len(self.data.query_to_idx)
                        response = all_responses[idx]
                        temp_instance.target_responses.append(response)
                        self.temp_results.data.append(temp_instance)

                    logger.info(f'Begin to evaluate the model responses')
                    # logger.debug(f'evaluator device: {self.evaluator.model.device}')
                    logger.debug(f'Finish evaluating the model responses')

                    mutator_instance.level = seed_instance.level + 1
                    mutator_instance.visited_num = 0

                    self.update(self.temp_results)
                    for instance in self.temp_results:
                        # self.data.attack_results.data.append(instance.copy())
                        example_idx = self.data.query_to_idx[instance.query]
                        
                        # Check if jailbreak_prompt is empty or invalid
                        if not instance.jailbreak_prompt or instance.jailbreak_prompt.strip() == "":
                            logger.warning(f"Empty jailbreak_prompt for query: {instance.query}")
                            final_query = instance.query  # Fallback to original query
                        elif '{query}' in instance.jailbreak_prompt:
                            final_query = instance.jailbreak_prompt.replace('{query}', instance.query)
                        else:
                            final_query = instance.jailbreak_prompt + " " + instance.query
                        
                        logger.debug(f'final_query: {final_query}\nresponse: {instance.target_responses[0] if instance.target_responses else "No response"}')

                        result = {
                            "example_idx": self.data.query_to_idx[instance.query],
                            "query": instance.query,
                            "final_query": final_query,
                            "response": instance.target_responses[0] if instance.target_responses else "",
                        }
                        if example_idx not in example_idx_to_results:
                            example_idx_to_results[example_idx] = result
                        else:
                            pass
                        # self.save(result)
                logger.info(f"Current iteration: {self.data.current_iteration}")
                
                with open(self.res_save_path, 'w') as f:
                    keys = sorted(example_idx_to_results.keys())
                    for key in keys:
                        f.write(json.dumps(example_idx_to_results[key], ensure_ascii=False) + '\n')
                

        except KeyboardInterrupt:
            logger.info("Fuzzing interrupted by user!")
        logger.info("Fuzzing finished!")


    def single_attack(self, instance: Example):
        """
        Perform an attack using a single query.
        """
        assert instance.jailbreak_prompt is not None, 'A jailbreak prompt must be provided'
        instance = instance.copy()
        instance.parents = []
        instance.children = []

        return_dataset = AttackDataset([])
        for i in range(self.config.energy):

            attack_dataset = AttackDataset([instance])
            new_dataset = []
            for sample in attack_dataset:
                mutated_instance_list = self.mutator.mutate(sample)
                new_dataset.extend(mutated_instance_list)
            instance = new_dataset[0]

            if instance.query is not None:
                # Check if jailbreak_prompt is empty or invalid
                if not instance.jailbreak_prompt or instance.jailbreak_prompt.strip() == "":
                    logger.warning(f"Empty jailbreak_prompt in single_attack for query: {instance.query}")
                    input_seed = instance.query  # Fallback to original query
                elif '{query}' in instance.jailbreak_prompt:
                    input_seed = instance.jailbreak_prompt.format(query=instance.query)
                else:
                    input_seed = instance.jailbreak_prompt + " " + instance.query
                    
                response = self.target_model.chat(input_seed)
                instance.target_responses.append(response)
            instance.parents = []
            instance.children = []
            return_dataset.data.append(instance)
        return return_dataset

    def is_stop(self):
        checks = [
            ('max_query', 'total_query'),
            ('max_jailbreak', 'total_jailbreak'),
            ('max_reject', 'total_reject'),
            ('max_iteration', 'current_iteration'),
        ]
        logger.debug(f'in is_stop check-> max_iteration: {self.config.max_iteration}, current_iteration: {self.data.current_iteration}')
        return any(getattr(self.config, max_attr) != -1 and getattr(self.data, curr_attr) >= getattr(self.config, max_attr) for
                   max_attr, curr_attr in checks)

    def update(self, dataset: AttackDataset):
        self.data.current_iteration += 1

        current_jailbreak = 0
        current_query = 0
        current_reject = 0
        for instance in dataset:
            current_jailbreak += instance.num_jailbreak
            current_query += instance.num_query
            current_reject += instance.num_reject

            self.data.total_jailbreak += instance.num_jailbreak
            self.data.total_query += instance.num_query
            self.data.total_reject += instance.num_reject

        self.data.current_jailbreak = current_jailbreak
        self.data.current_query = current_query
        self.data.current_reject = current_reject

        self.select_policy.update(dataset)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'GPTFuzzerManager':
        """Create GPTFuzzerManager from configuration dictionary."""
        return cls(**config)


def main():
    """Main function to run the GPTFuzzer attack."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        config_path = args.config_path or './configs/gptfuzzer.yaml'
        
        # Load configuration
        config_manager = ConfigManager(config_path=config_path)
        logger.info(f"Loaded configuration from: {config_path}")
        
        # Create and run attack manager
        attack_manager = GPTFuzzerManager.from_config(config=config_manager.config)
        attack_manager.attack()
        
    except Exception as e:
        logger.error(f"Failed to run GPTFuzzer attack: {e}")
        raise


if __name__ == "__main__":
    main()