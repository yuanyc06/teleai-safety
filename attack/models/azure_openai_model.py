from openai import AzureOpenAI, BadRequestError
import time
import logging

logger = logging.getLogger(__name__)

class AzureOpenAIModel:
    def __init__(self, model_name, base_url, api_key, api_version="2024-12-01-preview", generation_config=None):
        self.model_name = model_name
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=base_url,
        )
        self.conversation = get_conv_template('chatgpt')
        self.generation_config = generation_config if generation_config is not None else {}

        self.API_RETRY_SLEEP = 10
        self.API_ERROR_OUTPUT = "$ERROR$"
        self.API_QUERY_SLEEP = 0.5
        self.API_MAX_RETRY = 5
        self.API_TIMEOUT = 20
        self.API_LOGPROBS = True
        self.API_TOP_LOGPROBS = 20

    def set_system_message(self, system_message: str):
        self.conversation.system_message = system_message

    def generate(self, messages, clear_old_history=True, max_try=10, try_gap=5, gap_increase=5, **kwargs):
        if clear_old_history:
            self.conversation.messages = []

        if isinstance(messages, str):
            messages = [messages]

        if isinstance(messages[0], dict):
            if messages[0].get('role') == 'system':
                self.conversation.set_system_message(messages[0]['content'])
                messages = messages[1:]
            messages = [msg['content'] for msg in messages]

        for index, message in enumerate(messages):
            self.conversation.append_message(self.conversation.roles[index % 2], message)

        cur = 0
        temp_gen_config = self.generation_config.copy()
        if kwargs:
            temp_gen_config.update(kwargs)

        while cur < max_try:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.conversation.to_openai_api_messages(),
                    **temp_gen_config
                )

                logger.debug(f"[Prompt]\n{self.conversation.to_openai_api_messages()}")
                logger.debug(f"[Raw Response]\n{response}")

                choices = getattr(response, "choices", [])
                if not choices or not getattr(choices[0].message, "content", "").strip():
                    raise Exception("Empty or invalid response from AzureOpenAI")

                content = choices[0].message.content.strip()
                return content


            except BadRequestError as e:
                if "filtered due to the prompt triggering" in str(e):
                    logger.warning("Prompt was filtered by Azure content policy. Skipping this input.")
                    return "[Filtered by Content Policy]"
                else:
                    logger.error(f"[BadRequestError] {e}")
                    return "[BadRequestError]"

            except Exception as e:
                logger.error(f"[Retry {cur+1}/{max_try}] Failed to generate response from {self.model_name}: {e}")
                cur += 1
                if cur < max_try:
                    time.sleep(try_gap)
                    try_gap += gap_increase
                else:
                    logger.error("[Final Failure] Max retries reached. Returning fallback.")
                    return "[Failed to generate response]"

    def chat(self, messages, clear_old_history=True, max_try=5, try_gap=3, **kwargs):
        return self.generate(messages, clear_old_history, max_try, try_gap, **kwargs)

    def batch_chat(self, batch_messages, clear_old_history=True, max_try=5, try_gap=3, **kwargs):
        return [self.chat(messages, clear_old_history, max_try, try_gap, **kwargs) for messages in batch_messages]

    def get_response(self, prompts_list, max_n_tokens=None, no_template=False, gen_config={}):
        if isinstance(prompts_list[0], str):
            prompts_list = [[{'role': 'user', 'content': prompt}] for prompt in prompts_list]

        convs = prompts_list
        outputs = []
        for conv in convs:
            output = self.API_ERROR_OUTPUT
            for _ in range(self.API_MAX_RETRY):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=conv,
                        max_tokens=max_n_tokens,
                        **gen_config,
                        timeout=self.API_TIMEOUT,
                        logprobs=self.API_LOGPROBS,
                        top_logprobs=self.API_TOP_LOGPROBS,
                        seed=0,
                    )
                    response_logprobs = [
                        dict((response.choices[0].logprobs.content[i_token].top_logprobs[i_top_logprob].token,
                              response.choices[0].logprobs.content[i_token].top_logprobs[i_top_logprob].logprob)
                             for i_top_logprob in range(self.API_TOP_LOGPROBS))
                        for i_token in range(len(response.choices[0].logprobs.content))
                    ]
                    output = {
                        'text': response.choices[0].message.content,
                        'logprobs': response_logprobs,
                        'n_input_tokens': response.usage.prompt_tokens,
                        'n_output_tokens': response.usage.completion_tokens,
                    }
                    break
                except Exception as e:
                    logger.error(f"[get_response Retry] {type(e)} {e}")
                    time.sleep(self.API_RETRY_SLEEP)

                time.sleep(self.API_QUERY_SLEEP)

            outputs.append(output)
        return outputs
    
    def _extract_content_from_response(self, response):
        """
        尝试从 SDK 返回的 response 中提取文本内容，
        支持多种可能的字段路径并保证返回字符串或 None。
        """
        try:
            choices = getattr(response, "choices", None) or []
            if not choices:
                return None

            choice0 = choices[0]

            # Safe get message object (may be None)
            message_obj = getattr(choice0, "message", None)

            # 1) 常见： choice.message.content
            if message_obj is not None:
                content = getattr(message_obj, "content", None)
                if content is not None:
                    return str(content).strip()

            # 2) 退化：一些 SDK/版本可能直接使用 'text' 字段
            text = getattr(choice0, "text", None)
            if text:
                return str(text).strip()

            # 3) 流式或 delta 可能在 .delta.content
            delta = getattr(choice0, "delta", None)
            if delta:
                dcont = getattr(delta, "content", None)
                if dcont:
                    return str(dcont).strip()

            # 4) 最后尝试把 response 转成 str 但只作 debug 用（不作为正常输出）
            return None
        except Exception as e:
            logger.exception("Failed while extracting content from response: %s", e)
            return None



def get_conv_template(model_name: str):
    # if model_name.lower() == 'chatgpt':
    #     return ChatGPTConversation()
    # else:
    #     raise ValueError(f"Unknown conversation template: {model_name}")
    return ChatGPTConversation()


class ChatGPTConversation:
    def __init__(self):
        self.system_message = ""
        self.messages = []
        self.roles = ['user', 'assistant']

    def append_message(self, role, content):
        self.messages.append({'role': role, 'content': content})

    def update_last_message(self, content):
        if self.messages:
            self.messages[-1]['content'] = content

    def to_openai_api_messages(self):
        result = []
        if self.system_message:
            result.append({'role': 'system', 'content': self.system_message})
        result.extend(self.messages)
        return result

    def set_system_message(self, content):
        self.system_message = content
