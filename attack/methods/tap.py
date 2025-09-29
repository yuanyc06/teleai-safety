"""
This module implements a jailbreak method describe in the paper below.
This part of code is based on the code from the paper and EasyJailbreak.

Paper title: Tree of Attacks: Jailbreaking Black-Box LLMs Automatically
arXiv link: https://arxiv.org/abs/2312.02119
Source repository: https://github.com/RICommunity/TAP
Source repository: https://github.com/EasyJailbreak/EasyJailbreak
"""
import os
import sys
sys.path.append(os.getcwd())

import copy
from tqdm import tqdm
from dataclasses import dataclass, field, fields
from typing import List, Optional
from initialization import InitTemplates
from selection import DeleteOffTopic, ScoresSelection
from dataset import AttackDataset, Example
import logger
from utils import BaseAttackManager, ConfigManager, parse_arguments
from evaluation import PromptedLLMScorer, PatternScorer, HarmBenchScorer, LlamaGuard3Scorer
from mutation import *
from models import load_model, LocalModel, OpenAIModel, FORMATTER_REGISTRY, DefaultFormatter, AzureOpenAIModel
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class AttackConfig:
    data_path: str
    attack_model_type: str  # openai, azure, local
    attack_model_name: str
    attack_model_path: str
    target_model_type: str  # openai, azure,, local
    target_model_name: str
    target_model_path: str
    eval_model_type: str  # openai, azure, local
    eval_model_name: str
    eval_model_path: str
    api_key: str
    base_url: str
    azure_key: str
    azure_url: str
    grok_key: str
    grok_url: str
    evaluator_type: str
    evaluator_model_path: str
    tree_width: int = 10
    tree_depth: int = 10
    root_num: int = 1
    branching_factor: int = 4
    keep_last_n: int = 3
    max_n_attack_attempts: int = 5
    attack_max_n_tokens: int = 500
    attack_temperature: float = 1.0
    attack_top_p: float = 0.9
    target_max_n_tokens: int = 150
    target_temperature: float = 1.0
    target_top_p: float = 1.0
    judge_max_n_tokens: int = 10
    judge_temperature: float = 1.0
    devices: str = 'cuda:1'
    res_save_path: Optional[str] = None


@dataclass
class AttackData:
    attack_dataset: AttackDataset
    init_templates: List[str]
    batch: List[AttackDataset] = field(default_factory=list)
    current_jailbreak: int = 0
    iteration: int = 0
    example_idx: int = 0
    find_flag: bool = False


class TAPInit:
    def __init__(self, config: AttackConfig):
        self.config = config

        if config.data_path is None:
            attack_dataset = None
        else:
            attack_dataset = AttackDataset(config.data_path)

        init_templates = InitTemplates().get_templates('TAP', 1)
        self.data = AttackData(attack_dataset=attack_dataset, init_templates=init_templates)
        # self.device = torch.device(config.devices)

        self.attack_model = load_model(
            model_type=config.attack_model_type,
            model_name=config.attack_model_name,
            model_path=config.attack_model_path,
            config=config
        )
        
        self.target_model = load_model(
            model_type=config.target_model_type,
            model_name=config.target_model_name,
            model_path=config.target_model_path,
            config=config
        )
        
        self.eval_model = load_model(
            model_type=config.eval_model_type,
            model_name=config.eval_model_name,
            model_path=config.eval_model_path,
            config=config
        )


class TAPMutator:
    def __init__(self, attack_model, system_prompt, keep_last_n, branching_factor, max_n_attack_attempts):
        self.mutator = IntrospectGeneration(
            attack_model,
            system_prompt=system_prompt,
            keep_last_n=keep_last_n,
            branching_factor=branching_factor,
            max_n_attack_attempts=max_n_attack_attempts
        )

    def mutate(self, dataset):
        return self.mutator(dataset)


class TAPSelector:
    def __init__(self, attack_dataset, eval_model, tree_width):
        self.selector = ScoresSelection(attack_dataset, tree_width)
        self.constraintor = DeleteOffTopic(eval_model, tree_width)

    def select(self, dataset):
        return self.selector.select(dataset)

    def constraint(self, dataset):
        return self.constraintor(dataset)

class TAPEvaluator:
    def __init__(self, eval_model, evaluator_type, evaluator_model_path):
        self.evaluator = PromptedLLMScorer(eval_model, prompt_template = 'Mehrotra2023TAP')
        
        self.evaluator_type = evaluator_type
        if evaluator_type == "pattern":
            self.scorer = PatternScorer()
        elif evaluator_type == "harmbench":
            self.scorer = HarmBenchScorer(model_path=evaluator_model_path, tokenizer_path=evaluator_model_path)
        elif evaluator_type == "llamaguard3":
            self.scorer = LlamaGuard3Scorer(model_path=evaluator_model_path, tokenizer_path=evaluator_model_path)
        else:
            raise ValueError(f"Invalid evaluator type: {evaluator_type}")

    def evaluate(self, dataset):
        for example in dataset:
            example.eval_results.append(self.evaluator.score(example.query, example.target_responses[-1])['score'])

    def score(self, query, response):  
        is_success = self.scorer.score(query, response)['score']
        return is_success

class TAPManager(BaseAttackManager):
    """
    A class used to manage and execute TAP attacks.

    Parameters
    ----------
    data_path: str
        The path to the dataset used for the attack.
    attack_model_name: str
        The name of the attack model used for generating attack prompts.
    attack_model_path: str
        The path to load the attack model.
    target_model_name: str
        The name of the target model that will be attacked.
    target_model_path: str
        The path to load the target model.
    eval_model_name: str
        The name of the evaluation model used for scoring the attack's success.
    eval_model_path: str
        The path to load the evaluation model.
    openai_key: Optional[str]
        The OpenAI API key for accessing OpenAI models (if applicable).
    openai_url: Optional[str]
        The OpenAI API URL for accessing OpenAI models (if applicable).
    evaluator_type: str
        The type of the evaluation model used for deciding the attack's success.
    evaluator_model_path: str
        The path to load the evaluation model.
    tree_width: int
        The width of the tree used in the attack.
    tree_depth: int
        The depth of the tree used in the attack.
    root_num: int
        The number of trees or batch of a single instance.
    branching_factor: int
        The number of children nodes generated by a parent node during Branching.
    keep_last_n: int
        The number of rounds of dialogue to keep during Branching(mutation).
    max_n_attack_attempts: int
        The max number of attempts to generating a valid adversarial prompt of a branch.
    attack_max_n_tokens: int
        max_n_tokens of the target model.
    attack_temperature: float
        temperature of the attack model.
    attack_top_p: float
        top p of the attack_model.
    target_max_n_tokens: int
        max_n_tokens of the target model.
    target_temperature: float
        temperature of the target model.
    target_top_p: float
        top p of the target model.
    judge_max_n_tokens: int
        max_n_tokens of the evaluation model.
    judge_temperature: float
        temperature of the evaluation model.
    devices: str
        The device used for the attack.
    res_save_path: Optional[str]
        The path where the attack results will be saved.

    Methods
    -------
    single_attack(instance: AttackData):
        Performs a single attack on the given data instance using theTAP approach.
    
    attack(save_path='TAP_attack_result.jsonl'):
        Executes the TAP attack on a dataset, processing each example and saving the results.
    
    mutate(prompt: str, target: str):
        Mutates a given prompt and target using the TAP attack method, returning the modified prompt and target response.

    """
    def __init__(
        self,
        data_path="thu-coai/AISafetyLab_Datasets/harmbench_standard",
        attack_model_type="local",
        attack_model_name="vicuna_v1.1",
        attack_model_path="lmsys/vicuna-13b-v1.5",
        target_model_type="azure",
        target_model_name="vicuna_v1.1",
        target_model_path="lmsys/vicuna-13b-v1.5",
        eval_model_type="azure",
        eval_model_name="openai",
        eval_model_path="gpt-4o-mini",
        evaluator_type="harmbench",
        evaluator_model_path="cais/HarmBench-Llama-2-13b-cls",
        api_key='',
        base_url= '',
        azure_key= '',
        azure_url= '',
        grok_key = '',
        grok_url= '',
        tree_width=10,
        tree_depth=10,
        root_num=1,
        branching_factor=4,
        keep_last_n=3,
        max_n_attack_attempts=5,
        attack_max_n_tokens=1000,
        attack_temperature=1.0,
        attack_top_p=0.9,
        target_max_n_tokens=512,
        target_temperature=1.0,
        target_top_p=1.0,
        judge_max_n_tokens=10,
        judge_temperature=1.0,
        model_path="",
        template_path="",
        devices="cuda:auto",
        res_save_path="./results/tap_results.jsonl",
    ):
        super().__init__(res_save_path)
        _fields = fields(AttackConfig)
        local_vars = locals()
        _kwargs = {field.name: local_vars[field.name] for field in _fields}
        self.config = AttackConfig(**_kwargs)
        self.init = TAPInit(self.config)
        self.data = self.init.data

        self.attack_model = self.init.attack_model
        self.target_model = self.init.target_model
        self.eval_model = self.init.eval_model

        self.mutator = TAPMutator(
            self.attack_model,
            system_prompt=self.data.init_templates[0],
            keep_last_n=self.config.keep_last_n,
            branching_factor=self.config.branching_factor,
            max_n_attack_attempts=self.config.max_n_attack_attempts,
        )

        self.selector = TAPSelector(self.data.attack_dataset, self.eval_model, self.config.tree_width)
        self.evaluator = TAPEvaluator(self.eval_model, evaluator_type, evaluator_model_path)

        self.current_jailbreak = 0

        self.configure_models()

        if data_path is None:
            data_name = 'single-prompt'
        else:
            data_name = data_path.split("/")[-1][:-5]

    def configure_models(self):
        if not self.attack_model.generation_config:
            if isinstance(self.attack_model, OpenAIModel) or isinstance(self.attack_model, AzureOpenAIModel):
                self.attack_model.generation_config = {
                    'max_tokens': self.config.attack_max_n_tokens,
                    'temperature': self.config.attack_temperature,
                    'top_p': self.config.attack_top_p,
                }
            elif isinstance(self.attack_model, LocalModel):
                self.attack_model.generation_config = {
                    'max_new_tokens': self.config.attack_max_n_tokens,
                    'temperature': self.config.attack_temperature,
                    'do_sample': True,
                    'top_p': self.config.attack_top_p,
                    'eos_token_id': self.attack_model.tokenizer.eos_token_id,
                }
        if not self.eval_model.generation_config:
            if isinstance(self.eval_model, OpenAIModel) or isinstance(self.attack_model, AzureOpenAIModel):
                self.eval_model.generation_config = {
                    'max_tokens': self.config.judge_max_n_tokens,
                    'temperature': self.config.judge_temperature,
                }
            elif isinstance(self.eval_model, LocalModel):
                self.eval_model.generation_config = {
                    'do_sample': True,
                    'max_new_tokens': self.config.judge_max_n_tokens,
                    'temperature': self.config.judge_temperature,
                }

    def attack(self):
        logger.info("Jailbreak started!")
        for example_idx, example in enumerate(
            tqdm(self.data.attack_dataset, desc="Attacking...")
        ):
            self.data.example_idx = example_idx
            self.data.find_flag = False
            query = example.query
            print(f"QUERY:{'='*20}\n{example.query}")

            self.data.batch = [AttackDataset([copy.deepcopy(example)]) for _ in range(self.config.root_num)]

            for iteration in range(1, self.config.tree_depth + 1):
                self.data.iteration = iteration
                print(f"\n{'=' * 36}\nTree-depth is: {iteration}\n{'=' * 36}\n", flush=True)

                for i, stream in enumerate(self.data.batch):
                    print(f"BATCH:{i}")
                    new_dataset = stream

                    new_dataset = self.mutator.mutate(new_dataset)

                    if len(new_dataset) == 0:
                        print("New dataset is empty before constraint. Skipping iteration.")
                        continue

                    new_dataset = self.selector.constraint(new_dataset)

                    if len(new_dataset) == 0:
                        print("New dataset is empty after constraint. Skipping iteration.")
                        continue
                    

                    
                    
                    for ex in new_dataset:
                        try:
                            ex.target_responses = [ self.target_model.chat(ex.jailbreak_prompt) ]
                        except Exception as e:
                            logger.warning(f"Target model chat failed: {e}")
                            ex.target_responses = ["[GenerationFailed]"]

                    self.evaluator.evaluate(new_dataset)

                    new_dataset = self.selector.select(new_dataset)

                    self.data.batch[i] = new_dataset           

                    for ex in new_dataset:
                        self.log(
                            {
                                'example_idx': example_idx,
                                'Tree-depth': iteration,
                                'jailbreak_prompt': ex.jailbreak_prompt,
                                'query': ex.query,
                                'response': ex.target_responses,
                            },
                        )

                    for ex in new_dataset:
                        is_success = self.evaluator.score(ex.query, ex.target_responses[-1])
                        if is_success:
                            ex.eval_results[-1] = 100
                            self.data.find_flag = True
                            print("Found a jailbreak. Exiting.")
                            break

                    if self.data.find_flag:
                        new_example = max(new_dataset, key=lambda ex: ex.eval_results[-1])
                        new_example.eval_results = [1]
                        self.log(
                              {
                              'example_idx': example_idx,
                              'Tree-depth': iteration,
                              'success': True,
                              'jailbreak_prompt': new_example.jailbreak_prompt,
                              'query': new_example.query,
                              'response': new_example.target_responses,
                              },
                              save=True,
                        )
                        self.current_jailbreak += 1
                        break

            if self.data.iteration == self.config.tree_depth:
                if not new_dataset:
                    logger.warning("[Warning] new_dataset is empty at final iteration, skipping this example.")
                    continue  # 跳过当前样本，进入下一个 example_idx
                new_example = max(new_dataset, key=lambda ex: ex.eval_results[-1])
                new_example.eval_results = [0]

        asr = 100 * self.current_jailbreak / len(self.data.attack_dataset)
        print(f"ASR: {asr}%")
        logger.info("Jailbreak finished!")
        logger.info(
            'Jailbreak result saved at {}'.format(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config.res_save_path)
            )
        )

        if self.data.find_flag:
            return new_example.jailbreak_prompt
        else:
            new_example = max(new_dataset, key=lambda ex: ex.eval_results[-1])
            return new_example.jailbreak_prompt

    def mutate(self, prompt:str, target:str="Sure, here's"):
        example = Example(query=prompt, target=target)
        attack_dataset = AttackDataset([example])
        self.data.attack_dataset = attack_dataset
        results = self.attack()

        return results

      # 修复攻击主循环中的一些问题
    def fixed_attack_method(self):
      """
      修复后的攻击方法，确保与变异器接口正确对接
      """
      logger.info("Jailbreak started!")
      for example_idx, example in enumerate(tqdm(self.data.attack_dataset, desc="Attacking...")):
            self.data.example_idx = example_idx
            self.data.find_flag = False
            query = example.query
            print(f"QUERY:{'='*20}\n{example.query}")

            self.data.batch = [AttackDataset([copy.deepcopy(example)]) for _ in range(self.config.root_num)]

            for iteration in range(1, self.config.tree_depth + 1):
                  self.data.iteration = iteration
                  print(f"\n{'=' * 36}\nTree-depth is: {iteration}\n{'=' * 36}\n", flush=True)

                  for i, stream in enumerate(self.data.batch):
                        print(f"BATCH:{i}")
                  new_dataset = stream

                  # 变异步骤
                  try:
                        new_dataset = self.mutator.mutate(new_dataset)  # 现在接口匹配了
                  except Exception as e:
                        logger.error(f"变异步骤失败: {e}")
                        continue

                  if len(new_dataset) == 0:
                        print("变异后数据集为空，跳过约束步骤。")
                        continue

                  # 约束步骤
                  try:
                        new_dataset = self.selector.constraint(new_dataset)
                  except Exception as e:
                        logger.error(f"约束步骤失败: {e}")
                        continue

                  if len(new_dataset) == 0:
                        print("约束后数据集为空，跳过迭代。")
                        continue

                  # 获取目标模型响应
                  for ex in new_dataset:
                        try:
                              ex.target_responses = [self.target_model.chat(ex.jailbreak_prompt)]
                        except Exception as e:
                              logger.warning(f"目标模型聊天失败: {e}")
                              ex.target_responses = ["[GenerationFailed]"]

                  # 评估步骤
                  try:
                        self.evaluator.evaluate(new_dataset)
                  except Exception as e:
                        logger.error(f"评估步骤失败: {e}")
                        # 如果评估失败，给一个默认分数
                        for ex in new_dataset:
                              if not hasattr(ex, 'eval_results') or len(ex.eval_results) == 0:
                                    ex.eval_results = [0]

                  # 选择步骤
                  try:
                        new_dataset = self.selector.select(new_dataset)
                  except Exception as e:
                        logger.error(f"选择步骤失败: {e}")
                        continue

                  self.data.batch[i] = new_dataset

                  # 记录日志
                  for ex in new_dataset:
                        self.log({
                              'example_idx': example_idx,
                              'Tree-depth': iteration,
                              'jailbreak_prompt': ex.jailbreak_prompt,
                              'query': ex.query,
                              'response': ex.target_responses,
                        })

                  # 检查是否成功
                  for ex in new_dataset:
                        try:
                              is_success = self.evaluator.score(ex.query, ex.target_responses[-1])
                              if is_success:
                                    ex.eval_results[-1] = 100
                              self.data.find_flag = True
                              break
                        except Exception as e:
                              logger.error(f"Failed: {e}")
                              continue

                  new_example = max(new_dataset, key=lambda ex: ex.eval_results[-1] if ex.eval_results else 0)
                  new_example.eval_results = [1]
                  self.log({
                        'example_idx': example_idx,
                        'Tree-depth': iteration,
                        'jailbreak_prompt': new_example.jailbreak_prompt,
                        'query': new_example.query,
                        'response': new_example.target_responses,
                        }, save=True)
                  
            # 处理最终结果
            if self.data.iteration == self.config.tree_depth:
                  if not new_dataset:
                        logger.warning("[Warning] Skip.")
                        continue
                  new_example = max(new_dataset, key=lambda ex: ex.eval_results[-1] if ex.eval_results else 0)
                  new_example.eval_results = [0]

      logger.info("Jailbreak finished!")
      logger.info('Jailbreak result saved at {}'.format(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), self.config.res_save_path)
      ))

      if self.data.find_flag:
            return new_example.jailbreak_prompt
      else:
            new_example = max(new_dataset, key=lambda ex: ex.eval_results[-1] if ex.eval_results else 0)
            return new_example.jailbreak_prompt


def main():
    """Main function to run the Jailbroken attack."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        config_path = args.config_path or './configs/tap.yaml'
        
        # Load configuration
        config_manager = ConfigManager(config_path=config_path)
        logger.info(f"Loaded configuration from: {config_path}")
        
        # Create and run attack manager
        attack_manager = TAPManager.from_config(config=config_manager.config)
        attack_manager.fixed_attack_method()
        
    except Exception as e:
        logger.error(f"Failed to run Jailbroken attack: {e}")
        raise


if __name__ == "__main__":
    main()