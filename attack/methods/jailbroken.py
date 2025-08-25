"""
Jailbroken Attack Method
============================================
This Class achieves a jailbreak method describe in the paper below.

Paper title: Jailbroken: How Does LLM Safety Training Fail?
arXiv link: https://arxiv.org/pdf/2307.02483.pdf
Source repository: None
"""
import os
import sys
sys.path.append(os.getcwd())

from tqdm import tqdm
from dataset import AttackDataset
from dataclasses import dataclass
from logger import setup_logger
from models import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import torch.multiprocessing as mp
from fastchat.conversation import SeparatorStyle
import logger
import requests
from types import SimpleNamespace

from dataset import AttackDataset
from utils import BaseAttackManager, ConfigManager, parse_arguments
from evaluation import PatternScorer, HarmBenchScorer
from mutation import *
from models import load_model, LocalModel, OpenAIModel, FORMATTER_REGISTRY, DefaultFormatter
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

    
@dataclass
class AttackConfig:
    data_path: str
    system_prompt: str
    # Model settings
    attack_model_type: str # openai, azure, grok, local
    attack_model_path: str
    attack_model_name: str
    target_model_type: str # openai, azure, grok, local
    target_model_path: str
    target_model_name: str

    api_key: str
    base_url: str
    azure_key: str
    azure_url: str
    grok_key: str
    grok_url: str


    mutation_method: str='combination_3'
    attack_model_device: str=None
    target_model_device: str=None

@dataclass
class AttackData:
    attack_dataset: str

class JailbrokenEvaluator:
    def __init__(self):
        self.scorer = PatternScorer()
    
    def evaluate(self, query, response):
        
        is_success = self.scorer.score(query, response)['score']
        return is_success

class JailbrokenInit:
    def __init__(self, config: AttackConfig) -> None:
        attack_dataset = AttackDataset(config.data_path)
        self.data  = AttackData(attack_dataset = attack_dataset)
        
        print("Loading TargetModel...")
        self.target_model= load_model(
            model_type=config.target_model_type, 
            model_name=config.target_model_name, 
            model_path=config.target_model_path, 
            config=config
            )
        print("Loading AttackModel...")
        self.attack_model = load_model(model_type=config.attack_model_type, model_path=config.attack_model_path, model_name = config.attack_model_name, config=config)

class JailbrokenMutator:
    def __init__(self, config, init):
        self.config = config
        self.mutaitions = {
            'artificial':Artificial(),
            'base64':Base64(),
            'base64_input_only':Base64_input_only(),
            'base64_raw':Base64_raw(),
            'combination_1':Combination_1(),
            'combination_2':Combination_2(),
            'combination_3':Combination_3(),
            'disemvowel': Disemvowel(),
            'leetspeak': Leetspeak(),
            'rot13':Rot13(),
            'auto_payload_splitting':Auto_payload_splitting(init.attack_model),
            'auto_obfuscation':Auto_obfuscation(init.attack_model),
        }
        self.mutation = self.mutaitions[self.config.mutation_method]

    def mutate(self, query):
        mutated_queries = self.mutation.mutate(query)
        if isinstance(mutated_queries, str):
            mutated_queries = [mutated_queries]
        return mutated_queries

class JailbrokenManager(BaseAttackManager):

    def __init__(
        self,
        data_path="thu-coai/AISafetyLab_Datasets/harmbench_standard",
        system_prompt="",
        attack_model_type= 'local',
        attack_model_path="lmsys/vicuna-7b-v1.5",
        attack_model_name="vicuna_1.5",
        target_model_type= 'local',
        target_model_path="lmsys/vicuna-7b-v1.5",
        target_model_name="vicuna_1.5",
        api_key='',
        base_url='',
        azure_key='',
        azure_url='',
        grok_key='',   
        grok_url='', 
        mutation_method="combination_3",
        attack_model_device="cuda:0",
        target_model_device="cuda:1",
        res_save_path="results/jailbroken.jsonl",
        *args,
        **kwargs
    ):
        """
        A class used to manage and execute jailbreaking attacks on a target model.

        Parameters
        ----------
        data_path: str
            The path to load the jailbreaking prompts (default is 'thu-coai/AISafetyLab_Datasets/harmbench_standard').
        system_prompt: str
            The system prompt to the target_model (default is an empty string).
        attack_model_path: str
            The path to the model used for the attack (default is 'lmsys/vicuna-7b-v1.5').
        attack_model_name: str
            The name of the attack model (default is 'vicuna_1.5').
        model_path: str
            The path to the target model to attack (default is 'lmsys/vicuna-7b-v1.5').
        model_name: str
            The name of the target model (default is 'vicuna_1.5').
        mutation_method: str
            The method used for generating mutations in the jailbreaking attack (default is 'combination_3').
        attack_model_device: str
            The device on which the attack model is loaded (default is 'cuda:0').
        model_device: str
            The device on which the target model is loaded (default is 'cuda:1').
        res_save_path: str
            The path to save the results of the jailbreaking attack (default is 'results/jailbroken.jsonl').

        Methods
        -------
        mutate(): Mutate a single input string.
        attack(): Attack with defenders.
        """

        super().__init__(res_save_path)
        self.config = AttackConfig(data_path, system_prompt, attack_model_type, attack_model_path, attack_model_name, target_model_type, target_model_path, target_model_name, api_key, base_url, azure_key, azure_url, grok_key, grok_url,mutation_method)
        self.init = JailbrokenInit(self.config)
        self.data = self.init.data.attack_dataset
        self.evaluator = JailbrokenEvaluator()
        self.attack_model = self.init.attack_model
        self.target_model = self.init.target_model
        self.mutator = JailbrokenMutator(self.config, self.init)
        self.formatter = self._select_formatter()

    @classmethod
    def from_config(cls, config: dict, mutation_method=None):
        config = config.copy()
        mutation_method = mutation_method or config.pop("mutation_method", None)
        if mutation_method is None:
            raise ValueError("mutation_method must be specified.")
        return cls(mutation_method=mutation_method, **config)
    
    def _select_formatter(self):
        model_name = self.config.target_model_name.lower()
        system_prompt = system_prompt = getattr(self.config, "system_prompt", "You are a helpful assistant.")

        for keyword, formatter_cls in FORMATTER_REGISTRY.items():
            if keyword in model_name:
                return formatter_cls(system_prompt)
        
        return DefaultFormatter(system_prompt)
    def format_msgs(self, raw_query):
        return self.formatter.format(raw_query)

    def attack(self):
        for example_idx, example in enumerate(tqdm(self.data, desc="Attacking each example")):
            mutated_queries = self.mutator.mutate(example.query)
            for mutated_query in mutated_queries:
                final_input = self.format_msgs(mutated_query)
                if hasattr(self.target_model, "tokenizer"):
                    toks = self.target_model.tokenizer(final_input).input_ids
                    input_ids = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(self.target_model.model.device)
                    output_ids = self.target_model.generate(input_ids, batch=False)
                    response = self.target_model.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                else:
                    # OpenAI API 模型，直接调用 generate
                    response = self.target_model.chat(final_input)

                # is_success = self.evaluator.evaluate(example.query, response)
                save_data = {'example_idx': example_idx, 'query': example.query, 'final_query': mutated_query, 'response': response}
                self.log(save_data, save=True)

    def mutate(self, query):
        mutated_queries = self.mutator.mutate(query)
        if isinstance(mutated_queries, str):
            mutated_queries = [mutated_queries]
        return mutated_queries


def main():
    """Main function to run the Jailbroken attack."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        config_path = args.config_path or './configs/jailbroken.yaml'
        
        # Load configuration
        config_manager = ConfigManager(config_path=config_path)
        logger.info(f"Loaded configuration from: {config_path}")
        
        # Create and run attack manager
        attack_manager = JailbrokenManager.from_config(config=config_manager.config)
        attack_manager.attack()
        
    except Exception as e:
        logger.error(f"Failed to run Jailbroken attack: {e}")
        raise


if __name__ == "__main__":
    main()