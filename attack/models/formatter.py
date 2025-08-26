"""
Formatter classes for different models to format queries and system prompts.
"""

class BaseFormatter:
    def __init__(self, system_prompt=""):
        self.system_prompt = system_prompt

    def format(self, query: str) -> str:
        raise NotImplementedError("Each formatter must implement its own `format` method")


class VicunaFormatter(BaseFormatter):
    def format(self, query: str) -> str:
        prompt = f"{self.system_prompt or 'A chat between a human and an assistant.'}\n### Human: {query}\n### Assistant:"
        return prompt


class QwenFormatter(BaseFormatter):
    def format(self, query: str) -> str:
        system_str = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n" if self.system_prompt else ""
        user_msg = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        return system_str + user_msg


class LlamaFormatter(BaseFormatter):
    def format(self, query: str) -> str:
        if self.system_prompt:
            return f"{self.system_prompt}\nUser: {query}\nAssistant:"
        else:
            return f"User: {query}\nAssistant:"


class GrokFormatter(BaseFormatter):
    def format(self, query: str) -> str:
        return f"{self.system_prompt}\n{query}" if self.system_prompt else query


class DefaultFormatter(BaseFormatter):
    def format(self, query: str) -> str:
        return f"{self.system_prompt}\n{query}" if self.system_prompt else query


# 注册表：模型关键词 => 格式化器类
FORMATTER_REGISTRY = {
    "vicuna": VicunaFormatter,
    "qwen": QwenFormatter,
    "deepseek": QwenFormatter,
    "llama": LlamaFormatter,
    "grok": GrokFormatter,
}
