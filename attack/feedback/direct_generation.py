from models import LocalModel, OpenAIModel, AzureOpenAIModel
import json 
from openai import AzureOpenAI

# def generate(object, messages, input_field_name='input_ids', **kwargs):
    
#     object.conversation.messages = []
#     if isinstance(messages, str):
#         messages = [messages]
#     for index, message in enumerate(messages):
#         object.conversation.append_message(object.conversation.roles[index % 2], message)

#     if isinstance(object, LocalModel):
#         if object.conversation.roles[-1] not in object.conversation.get_prompt():
#             object.conversation.append_message(object.conversation.roles[-1], None)
#         prompt = object.conversation.get_prompt()

#         input_ids = object.tokenizer(prompt,
#                                     return_tensors='pt',
#                                     add_special_tokens=False).input_ids.to(object.model.device.index)
#         input_length = len(input_ids[0])

#         kwargs.update({input_field_name: input_ids})
#         output_ids = object.model.generate(**kwargs, **object.generation_config)
#         output = object.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

#     elif isinstance(object, OpenAIModel):
#         # response = object.client.chat.completions.create(
#         #     model=object.model_name,
#         #     messages=object.conversation.to_openai_api_messages(),
#         #     **kwargs,
#         #     **object.generation_config
#         # )
        
#         # output = response.choices[0].message.content
#         try:
#             response = object.client.chat.completions.create(
#                 model=object.model_name,
#                 messages=object.conversation.to_openai_api_messages(),
#                 **kwargs,
#                 **object.generation_config
#             )

#             print("[DEBUG] Raw OpenAI response:", response)

#             # 如果是字符串（代理平台很多是这种）
#             if isinstance(response, str):
#                 try:
#                     parsed = json.loads(response)
#                     print("[DEBUG] Parsed JSON:", parsed)
#                     output = parsed["choices"][0]["message"]["content"]
#                 except Exception as e:
#                     print("[ERROR] Failed to parse OpenAI string response as JSON:", e)
#                     output = response  # fallback
#             # 如果是标准 SDK 返回对象
#             elif hasattr(response, "choices"):
#                 output = response.choices[0].message.content
#             else:
#                 print("[ERROR] Unexpected response type:", type(response))
#                 output = str(response)

#         except Exception as e:
#             print(f"[ERROR] OpenAIModel generate failed with exception: {e}")
#             output = ""
#     elif isinstance(object, AzureOpenAIModel):
#         try:
#             response = object.client.chat.completions.create(
#                 model=object.model_name,
#                 messages=object.conversation.to_openai_api_messages(),
#                 **kwargs,
#                 **object.generation_config
#             )
            
#             output = response.choices[0].message.content
#             print("AzureOpenAIModel response:",output)
#         except Exception as e:
#             print(f"[ERROR] OpenAIModel generate failed with exception: {e}")
#             output = "The response was filtered due to the prompt triggering Azure OpenAI's content management policy."
        

#     else:
#         raise ValueError("Unknown model type passed to generate")

                

#     return output



def generate(object, messages, input_field_name='input_ids', AzureOpenAI_name=None, **kwargs):
    # 清空历史对话记录：兼容不同模型
    if hasattr(object, "conversation") and hasattr(object.conversation, "messages"):
        object.conversation.messages = []
    elif hasattr(object, "history"):
        object.history = []
    elif hasattr(object, "chat_history"):
        object.chat_history = []
    else:
        print("[INFO] No conversation history attribute found in model. Skipping reset.")

    # 统一消息格式
    if isinstance(messages, str):
        messages = [messages]

    for index, message in enumerate(messages):
        if hasattr(object, "conversation") and hasattr(object.conversation, "append_message"):
            object.conversation.append_message(object.conversation.roles[index % 2], message)
        elif hasattr(object, "history"):
            role = "user" if index % 2 == 0 else "assistant"
            object.history.append({"role": role, "content": message})
        elif hasattr(object, "chat_history"):
            role = "user" if index % 2 == 0 else "assistant"
            object.chat_history.append({"role": role, "content": message})
        else:
            print("[ERROR] Unknown message appending method. Please check your model.")

    # LocalModel 推理逻辑
    if isinstance(object, LocalModel):
        # —— 新增：如果 messages 就是个字符串，直接拿来当 prompt
        if isinstance(messages, str):
            prompt = messages

        # —— 否则继续走原有的各类对话接口
        elif hasattr(object, "conversation") and hasattr(object.conversation, "get_prompt"):
            if object.conversation.roles[-1] not in object.conversation.get_prompt():
                object.conversation.append_message(object.conversation.roles[-1], None)
            prompt = object.conversation.get_prompt()

        elif hasattr(object, "build_prompt"):
            prompt = object.build_prompt()

        elif hasattr(object, "history") and isinstance(object.history, list):
            prompt = ""
            for msg in object.history:
                prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
            prompt += "Assistant:"

        elif hasattr(object, "messages") and isinstance(object.messages, list):
            prompt = "\n".join(object.messages) + "\nAssistant:"

        elif hasattr(object, "system_prompt") and hasattr(object, "history"):
            prompt = object.system_prompt + "\n"
            for msg in object.history:
                prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
            prompt += "Assistant:"

        else:
            prompt = messages  # fallback

        # ✅ 安全处理空 prompt
        if not isinstance(prompt, str) or not prompt.strip():
            print(f"[WARNING] Empty prompt detected. Messages: {messages}")
            return "[NO RESPONSE]"

        # ✅ tokenizer 编码并确保 input_ids 不为空
        input_ids = object.tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=False
        ).input_ids.to(object.model.device.index)

        if input_ids.shape[1] == 0:
            print(f"[WARNING] Tokenizer returned empty input_ids for prompt: {repr(prompt)}")
            return "[NO RESPONSE]"

        input_length = input_ids.shape[1]

        kwargs.update({input_field_name: input_ids})
        try:
            output_ids = object.model.generate(**kwargs, **object.generation_config)
            output = object.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
            return output.strip() if output.strip() else "[NO RESPONSE]"
        except Exception as e:
            print(f"[ERROR] Generation failed. Prompt: {repr(prompt)} | Error: {e}")
            return "[GENERATION ERROR]"

    # if isinstance(object, LocalModel):
    #     # —— 新增：如果 messages 就是个字符串，直接拿来当 prompt
    #     if isinstance(messages, str):
    #         prompt = messages

    #     # —— 否则继续走原有的各类对话接口
    #     elif hasattr(object, "conversation") and hasattr(object.conversation, "get_prompt"):
    #         if object.conversation.roles[-1] not in object.conversation.get_prompt():
    #             object.conversation.append_message(object.conversation.roles[-1], None)
    #         prompt = object.conversation.get_prompt()

    #     elif hasattr(object, "build_prompt"):
    #         prompt = object.build_prompt()

    #     elif hasattr(object, "history") and isinstance(object.history, list):
    #         prompt = ""
    #         for msg in object.history:
    #             prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    #         prompt += "Assistant:"

    #     elif hasattr(object, "messages") and isinstance(object.messages, list):
    #         prompt = "\n".join(object.messages) + "\nAssistant:"

    #     elif hasattr(object, "system_prompt") and hasattr(object, "history"):
    #         prompt = object.system_prompt + "\n"
    #         for msg in object.history:
    #             prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    #         prompt += "Assistant:"

    #     else:
    #         # 不再抛错，直接回退到原始输入
    #         prompt = messages  # messages 肯定是个字符串或列表
            
    #     input_ids = object.tokenizer(prompt,
    #                                  return_tensors='pt',
    #                                  add_special_tokens=False).input_ids.to(object.model.device.index)
    #     input_length = len(input_ids[0])

    #     kwargs.update({input_field_name: input_ids})
    #     output_ids = object.model.generate(**kwargs, **object.generation_config)
    #     output = object.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

    # OpenAI Model 调用逻辑
    elif isinstance(object, OpenAIModel):
        try:
            response = object.client.chat.completions.create(
                model=object.model_name,
                messages=object.conversation.to_openai_api_messages(),
                **kwargs,
                **object.generation_config
            )

            # print("[DEBUG] Raw OpenAI response:", response)

            if isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    # print("[DEBUG] Parsed JSON:", parsed)
                    output = parsed["choices"][0]["message"]["content"]
                except Exception as e:
                    print("[ERROR] Failed to parse OpenAI string response as JSON:", e)
                    output = response
            elif hasattr(response, "choices"):
                output = response.choices[0].message.content
            else:
                print("[ERROR] Unexpected response type:", type(response))
                output = str(response)

        except Exception as e:
            print(f"[ERROR] OpenAIModel generate failed with exception: {e}")
            output = ""
            
    # Azure Model 调用逻辑
    elif isinstance(object, AzureOpenAI):
        try:
            response = object.client.chat.completions.create(
                model=AzureOpenAI_name,
                # messages=object.conversation.to_openai_api_messages(),
                messages=[
                    # {"role": "system", "content": object.sys_prompt},
                    {"role": "user", "content": message}
                ],
                **kwargs,
                **object.generation_config
            )

            # print("[DEBUG] Raw OpenAI response:", response)

            if isinstance(response, str):
                try:
                    parsed = json.loads(response)
                    # print("[DEBUG] Parsed JSON:", parsed)
                    output = parsed["choices"][0]["message"]["content"]
                except Exception as e:
                    print("[ERROR] Failed to parse OpenAI string response as JSON:", e)
                    output = response
            elif hasattr(response, "choices"):
                output = response.choices[0].message.content
            else:
                print("[ERROR] Unexpected response type:", type(response))
                output = str(response)

        except Exception as e:
            print(f"[ERROR] OpenAIModel generate failed with exception: {e}")
            output = ""

    else:
        raise ValueError("Unknown model type passed to generate")

    return output

