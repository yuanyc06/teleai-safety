# SCAV核心攻击流程集成AISafetyLab
import os
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional
from models import AzureOpenAIModel
import requests


        
class ScavConfig:
    def __init__(self, **kwargs):
        self.attacker_cls = kwargs.get('attacker_cls', 'ScavAttacker')
        self.attacker_name = kwargs.get('attacker_name', 'Scav')
        self.target_llm_name = kwargs.get('target_llm_name', 'gpt-4.1')
        self.api_key = kwargs.get('api_key', None)
        self.base_url = kwargs.get('base_url', None)
        self.dataset_path = kwargs.get('dataset_path', None)
        self.res_save_path = kwargs.get('res_save_path', None)
        self.prompt_file = kwargs.get('prompt_file', None)
        self.max_round = int(kwargs.get('max_round', 1))
        self.optimzed_file_8b = kwargs.get('optimzed_file_8b', None)
        self.optimzed_file_70b = kwargs.get('optimzed_file_70b', None)

# --- ScavPromptOptimizer ----------------------------------------------------
class ScavPromptOptimizer:
    def __init__(self, prompt_file: str, max_round: int = 1):
        self.prompt_file = prompt_file
        self.max_round = max_round
        self.df = pd.read_csv(prompt_file, header=None)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.strip().lower().split())   # 去空格+小写

    def _lookup(self, query: str) -> List[str]:
        norm_query = self._normalize(query)
        # 1) 精确匹配（忽略大小写和多余空格）
        exact = self.df[self.df[0].astype(str).apply(self._normalize) == norm_query]
        if not exact.empty:
            return exact[1].astype(str).tolist()

        # 2) 包含匹配
        contains = self.df[self.df[0].astype(str).apply(self._normalize).apply(lambda x: x in norm_query)]
        if not contains.empty:
            return contains[1].astype(str).tolist()

        # 3) 没命中 -> 返回原 query 本身
        return [query]

    def optimize(self, query: str) -> List[str]:
        prompts = [query]
        for _ in range(self.max_round):
            new_prompts = []
            for p in prompts:
                new_prompts.extend(self._lookup(p))
            prompts = list(dict.fromkeys(new_prompts))  # 去重保序
        return prompts


# --- ScavManager ------------------------------------------------------------
class ScavManager:
    def __init__(self, config: ScavConfig):
        self.config = config

        # 选择优化指令表
        prompt_file = config.prompt_file
        if not prompt_file:
            if "8b" in config.target_llm_name.lower() and config.optimzed_file_8b:
                prompt_file = config.optimzed_file_8b
            else:
                prompt_file = config.optimzed_file_70b

        if not prompt_file or not os.path.exists(prompt_file):
            raise FileNotFoundError(f"SCAV 优化指令表未找到: {prompt_file}")

        self.optimizer = ScavPromptOptimizer(prompt_file, max_round=config.max_round)

        # 目标模型(可选)
        if "grok" in config.target_llm_name.lower():
            self.target_model = Grok3Model(
                model_name=config.target_llm_name,
                base_url=config.base_url,
                api_key=config.api_key
            ) if config.api_key and config.base_url else None
        else:
                # 使用 Azure OpenAI 模型
            self.target_model = AzureOpenAIModel(
                model_name=config.target_llm_name,
                base_url=config.base_url,
                api_key=config.api_key
            ) if config.api_key and config.base_url else None

        # 结果文件
        self.res_save_path = config.res_save_path
        if self.res_save_path:
            os.makedirs(os.path.dirname(self.res_save_path), exist_ok=True)

    def attack(self, dataset: List[Dict]):
        results = []
        f_out = open(self.res_save_path, "w", encoding="utf-8") if self.res_save_path else None

        for idx, item in enumerate(tqdm(dataset, desc="SCAV Attacking")):
            query = item.get("query") or item.get("question") or item.get("content")
            if not query:
                continue

            optimized_prompts = self.optimizer.optimize(query)

            for opt_idx, opt_prompt in enumerate(optimized_prompts):
                # 如果未命中优化表，opt_prompt 会等于原 query；用 "N/A" 标记
                if opt_prompt == query:
                    opt_prompt_tag = "N/A"
                else:
                    opt_prompt_tag = opt_prompt

                # 请求目标模型
                if self.target_model:
                    if "grok" in self.target_model.model_name.lower():
                        response = self.target_model.chat(opt_prompt)
                    else:
                        try:
                            response = self.target_model.chat(
                                messages=[{"role": "user", "content": opt_prompt}]
                            )
                        except Exception as e:
                            response = f"[Error] {e}"
                else:
                    response = None

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
                    f_out.flush()           # 实时写盘,防 0B

        if f_out:
            f_out.close()
        return results

    def single_attack(self, query: str):
        optimized_prompts = self.optimizer.optimize(query)
        responses = []
        for opt_prompt in optimized_prompts:
            if self.target_model:
                try:
                    response = self.target_model.chat(messages=[{"role": "user", "content": opt_prompt}])
                except Exception as e:
                    response = f"[Error] {e}"
            else:
                response = None
            responses.append({"optimized_prompt": opt_prompt, "response": response})
        return responses
