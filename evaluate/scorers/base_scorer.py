from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
import re
from .debate_prompt_const import SAFETY_DEBATE_PROMPTS, debate_history_dict,FORMAT_STRING_CC, FORMAT_STRING, zhang2024shieldlm, harm_bench_prompt

from dataclasses import dataclass
import yaml
import torch

@dataclass
class BaseScorerConfig:
    model_path: str = None
    device: str = None

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            cfg = {}
        return cls(**cfg)


class BaseScorer:
    def __init__(self):
        pass

    def score(self, query: str, response: str):
        raise NotImplementedError

@dataclass
class PromptedLLMScorerConfig:
    debate: bool = False
    role: str = "None"

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            cfg = {}
        return cls(**cfg)

class PromptedLLMScorer(BaseScorer):
    def __init__(self, config: PromptedLLMScorerConfig = PromptedLLMScorerConfig()):
        self.config = config
        self.debate = config.debate
        self.role = config.role
        self.prompt_template = SAFETY_DEBATE_PROMPTS[self.config.role] if self.config.debate and self.config.role in SAFETY_DEBATE_PROMPTS else zhang2024shieldlm

    def chat(self, text: str):
        raise NotImplementedError

    def extract_res(self, text):
        """
        Extract [Answer] and [Analysis] from model output.
        """
        if self.role == "Counterargument Critic":
            answer, score = None, None
        else:
            answer_match = re.search(r"\[Answer\]\s*(.+)", text)
            answer = answer_match.group(1).strip() if answer_match else ""
            score = 1 if answer.lower() == 'unsafe' else 0
        analysis_match = re.search(r"\[Analysis\]\s*(.+)", text, re.DOTALL)
        analysis = analysis_match.group(1).strip() if analysis_match else ""
        return {"score": score, "answer": answer, "analysis": analysis, "output": text}

    def score(self, query: str, response: str, chat_history=None, *args, **kwargs):
        """
        Score response using the prompt template and model chat.
        """
        if self.debate:
            if chat_history is None:
                history_str = ""
                prompt_template = self.prompt_template.format(debate_prompt='')
            else:
                history_str = chat_history
                prompt_template = self.prompt_template.format(debate_prompt=debate_history_dict[self.role]) if self.role in debate_history_dict else self.prompt_template
            if self.role == "Counterargument Critic":
                prompt = prompt_template + FORMAT_STRING_CC.format(query=query, response=response, DEBATE_HISTORY=("[Debate History]:\n\n" + history_str))
            else:
                prompt = prompt_template + FORMAT_STRING.format(query=query, response=response, DEBATE_HISTORY=("[Debate History]:\n\n" + history_str) if chat_history is not None else "")
        else:
            prompt = self.prompt_template.format(query=query, response=response)
        model_response = self.chat(prompt)
        return self.extract_res(model_response) if model_response is not None else {"score": None, 'answer': None, 'analysis': None, 'output': None}

@dataclass
class HuggingFaceScorerConfig(PromptedLLMScorerConfig):
    model_path: str = None
    device: str = "cuda:0"
    tokenizer = None
    model = None
    debate: bool = False
    role: str = "None"
    max_new_tokens: int = 40
    
class HuggingFacePromptedLLMScorer(PromptedLLMScorer):
    def __init__(self, config: HuggingFaceScorerConfig = HuggingFaceScorerConfig()):
        super().__init__(config=config)
        self.model_path = config.model_path
        self.device = config.device if torch.cuda.is_available() else "cpu"
        self.tokenizer = config.tokenizer
        self.model = config.model
        self.max_new_tokens = config.max_new_tokens
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map=self.device)
        logger.debug(f"{self.__class__.__name__} model loaded from {self.model_path}, device: {self.device}")

    def chat(self, text: str):
        messages = [{"role": "user", "content": text}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        return response

@dataclass
class APIPromptedLLMScorerConfig(PromptedLLMScorerConfig):
    debate: bool = False
    role: str = "None"

class APIPromptedLLMScorer(PromptedLLMScorer):
    def __init__(self, config: APIPromptedLLMScorerConfig = APIPromptedLLMScorerConfig(), api_func: callable = None):
        super().__init__(config=config)
        self.api_func = api_func

    def chat(self, text: str):
        try:
            response = self.api_func(text)
            return response
        except Exception:
            return None
