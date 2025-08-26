import os
import sys
sys.path.append(os.getcwd())

from tqdm import tqdm
from dataset import AttackDataset
import pandas as pd

from dataclasses import dataclass
from logger import setup_logger
from models import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch.multiprocessing as mp
from fastchat.conversation import SeparatorStyle
import logger
from typing import List, Dict

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
    target_model_type: str
    target_model_path: str
    target_model_name: str

    api_key: str
    base_url: str
    azure_key: str
    azure_url: str
    grok_key: str
    grok_url: str

    target_model_device: str = None

    prompt_file: str = None
    optimized_file_8b: str = None
    optimized_file_70b: str = None
    max_round: int = 1
    res_save_path: str = None

@dataclass
class AttackData:
    attack_dataset: str


class ScavPromptOptimizer:
    def __init__(self, prompt_file: str, max_round: int = 1):
        self.prompt_file = prompt_file
        self.max_round = max_round
        self.df = pd.read_csv(prompt_file, header=None)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _lookup(self, query: str) -> List[str]:
        norm_query = self._normalize(query)
        exact = self.df[self.df[0].astype(str).apply(self._normalize) == norm_query]
        if not exact.empty:
            return exact[1].astype(str).tolist()

        contains = self.df[self.df[0].astype(str).apply(self._normalize).apply(lambda x: x in norm_query)]
        if not contains.empty:
            return contains[1].astype(str).tolist()

        return [query]

    def optimize(self, query: str) -> List[str]:
        prompts = [query]
        for _ in range(self.max_round):
            new_prompts = []
            for p in prompts:
                new_prompts.extend(self._lookup(p))
            prompts = list(dict.fromkeys(new_prompts))
        return prompts


class ScavInit:
    def __init__(self, config: AttackConfig):
        self.config = config
        # 加载目标模型（统一入口）

        self.target_model = load_model(
            model_type=config.target_model_type,
            model_path=config.target_model_path,
            model_name=config.target_model_name,
            config=config
        )

        # 选择优化指令表
        prompt_file = config.prompt_file
        if not prompt_file:
            if "8b" in config.target_model_name.lower() and config.optimized_file_8b:
                prompt_file = config.optimized_file_8b
            else:
                prompt_file = config.optimized_file_70b

        if not prompt_file or not os.path.exists(prompt_file):
            raise FileNotFoundError(f"SCAV 优化指令表未找到: {prompt_file}")

        self.optimizer = ScavPromptOptimizer(prompt_file, max_round=config.max_round)

# -------------------------
# Manager
# -------------------------
class ScavManager(ScavInit):
    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            config = AttackConfig(**config)
        return cls(config)

    def attack(self):
        results = []
        res_path = self.config.res_save_path
        f_out = open(res_path, "w", encoding="utf-8") if res_path else None
        attack_dataset = AttackDataset(self.config.data_path)
        self.data  = AttackData(attack_dataset = attack_dataset)
        for idx, item in enumerate(tqdm(attack_dataset, desc="SCAV Attacking")):
            query = item.get("query") or item.get("question") or item.get("content")
            if not query:
                continue

            optimized_prompts = self.optimizer.optimize(query)

            for opt_idx, opt_prompt in enumerate(optimized_prompts):
                opt_prompt_tag = "N/A" if opt_prompt == query else opt_prompt

                try:
                    response = self.target_model.chat(opt_prompt)
                except Exception as e:
                    response = f"[Error] {e}"

                record = {
                    "index": idx,
                    "opt_index": opt_idx,
                    "query": query,
                    "optimized_prompt": opt_prompt_tag,
                    "response": response
                }
                results.append(record)

                if f_out:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()

        if f_out:
            f_out.close()
        return results

# -------------------------
# Main
# -------------------------
def main():
    try:
        args = parse_arguments()
        config_path = args.config_path or './configs/scav.yaml'
        config_manager = ConfigManager(config_path=config_path)
        logger.info(f"Loaded configuration from: {config_path}")

        attack_manager = ScavManager.from_config(config=config_manager.config)
        attack_manager.attack()

    except Exception as e:
        logger.error(f"Failed to run Scav attack: {e}")
        raise

if __name__ == "__main__":
    main()