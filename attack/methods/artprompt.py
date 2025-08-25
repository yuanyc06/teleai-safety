import os
import sys
sys.path.append(os.getcwd())

import json
import logging
from tqdm import tqdm
from typing import List, Dict
from dataclasses import dataclass

from dataset import AttackDataset
import logger
from models import load_model
from utils import ConfigManager, parse_arguments

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------------- Config -------------------------
@dataclass
class AttackConfig:
    data_path: str
    system_prompt: str

    # 模型相关
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

    # 文件路径
    prompt_template_path: str = None
    ascii_dict_path: str = None
    res_save_path: str = None

# ------------------------- Rewriter -------------------------
class ArtPromptRewriter:
    def __init__(self, config: AttackConfig):
        self.config = config
        # 加载 prompt 模板
        with open(config.prompt_template_path, "r", encoding="utf-8") as f:
            self.sys_prompt = f.read()
        # 加载 ASCII 字典
        ascii_path = config.ascii_dict_path or "/gemini/code/AISafetyLab-main/aisafetylab/attack/attackers/data/icl/artprompt/vitc-s.json"
        with open(ascii_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.ascii_dict = {item["text"]: item["art_text"] for item in raw}

    def _ascii_of(self, word: str) -> str:
        lines_per_char = self.ascii_dict["A"].count("\n")
        rows = [""] * lines_per_char
        for ch in word.upper():
            art = self.ascii_dict.get(ch)
            if not art:
                continue
            cols = art.split("\n")
            for i in range(lines_per_char):
                rows[i] += cols[i] + "  "
        return "\n" + "\n".join(rows) + "\n"

    def _parse(self, resp: str):
        mw, mi = [], ""
        if "### Masked words:" in resp:
            seg = resp.split("### Masked words:")[1].split("###")[0]
            mw = [w.strip().strip("'").strip('"') for w in seg[seg.find("[")+1:seg.find("]")].split(",")]
        if "### Masked instruction:" in resp:
            mi = resp.split("### Masked instruction:")[1].strip()
        return mw, mi

    def _inject_ascii(self, words: List[str], instr: str):
        ascii_arts = [self._ascii_of(w) for w in words]
        if len(ascii_arts) == 1:
            return instr.replace("[MASK]", ascii_arts[0])
        for i, art in enumerate(ascii_arts, 1):
            instr = instr.replace(f"[MASK{i}]", art)
        return instr

    def rewrite(self, target_model, query: str) -> str:
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": query},
        ]
        for attempt in range(5):
            try:
                content = target_model.chat(messages)
                masked_words, masked_instr = self._parse(content)
                if masked_instr:
                    return self._inject_ascii(masked_words, masked_instr)
                return f"Instruction parsing failed for query: {query}"
            except Exception as e:
                fallback_ascii = self._ascii_of("CENSORED")
                return f"Please follow the instruction:\n{fallback_ascii}\n"
        return f"Prompt generation failed after retries for: {query}"

# ------------------------- Init -------------------------
class ArtPromptInit:
    def __init__(self, config: AttackConfig):
        self.config = config
        self.target_model = load_model(
            model_type=config.target_model_type,
            model_path=config.target_model_path,
            model_name=config.target_model_name,
            config=config
        )
        self.rewriter = ArtPromptRewriter(config)

# ------------------------- Manager -------------------------
class ArtPromptManager(ArtPromptInit):
    @classmethod
    def from_config(cls, config):
        if isinstance(config, dict):
            config = AttackConfig(**config)
        return cls(config)

    def attack(self):
        results = []
        f_out = open(self.config.res_save_path, "w", encoding="utf-8") if self.config.res_save_path else None
        dataset = AttackDataset(self.config.data_path)
        for idx, item in enumerate(tqdm(dataset, desc="ArtPrompt Attacking")):
            query = item.get("query") or item.get("question") or item.get("content")
            if not query:
                continue
            art_prompt = self.rewriter.rewrite(self.target_model, query)
            try:
                answer = self.target_model.chat(art_prompt)
            except Exception as e:
                answer = f"[Error getting response] {str(e)}"
            record = {
                "index": idx,
                "query": query,
                "artprompt": art_prompt,
                "response": answer
            }
            results.append(record)
            if f_out:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
        if f_out:
            f_out.close()
        return results

# ------------------------- Main -------------------------
def main():
    args = parse_arguments()
    config_path = args.config_path or './configs/artprompt.yaml'
    config_manager = ConfigManager(config_path=config_path)
    logger.info(f"Loaded configuration from: {config_path}")
    attack_manager = ArtPromptManager.from_config(config_manager.config)
    attack_manager.attack()

if __name__ == "__main__":
    main()
