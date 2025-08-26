from utils import random_topk_substitute
from utils import batch_tidify
import torch
from types import SimpleNamespace
class GradientSubstituter:
    @staticmethod
    def sample_control(control_toks, grad, batch_size, topk=256, temp=1):
        # 改进的采样策略
        if temp != 1:
            # 使用温度缩放来调整梯度分布
            scaled_grad = grad / temp
        else:
            scaled_grad = grad
            
        top_indices = (-scaled_grad).topk(topk, dim=1).indices
        control_toks = control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)
        
        # 改进位置选择策略 - 增加随机性
        if batch_size <= len(control_toks):
            # 随机选择要替换的位置
            new_token_pos = torch.randperm(len(control_toks), device=grad.device)[:batch_size]
        else:
            # 如果batch_size更大，允许重复位置
            new_token_pos = torch.randint(0, len(control_toks), (batch_size,), device=grad.device)
            
        # 使用概率采样而不是完全随机
        if temp != 1:
            # 使用softmax采样而不是uniform采样
            probs = torch.softmax((-scaled_grad[new_token_pos]) / temp, dim=1)
            sampled_indices = torch.multinomial(probs, 1)
        else:
            sampled_indices = torch.randint(0, topk, (batch_size, 1), device=grad.device)
            
        new_token_val = torch.gather(top_indices[new_token_pos], 1, sampled_indices)
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks
    @staticmethod
    def substitute_adversarial_tokens_based_on_gradient(tokenizer, prompt_with_output_encoded, adv_indices, target_length, one_hot, prefix_separator, suffix_separator, KeepBest,replace_size, K, beam_size ):
        replace_size, K, beam_size = replace_size, K, beam_size
        one_hot = one_hot

        original = prompt_with_output_encoded[..., adv_indices]
        
        variants_tensor = random_topk_substitute(
            -one_hot.grad, original,
            num_sub_positions=replace_size,
            topK=K, beam_size=beam_size
        )

        if KeepBest:
            variants_tensor = torch.cat((prompt_with_output_encoded[..., adv_indices], variants_tensor), dim=0)

        variants_tensor = batch_tidify(
            variants_tensor, target_length,
            tokenizer, prefix_separator, suffix_separator
        )

        new_prompt_with_output_encoded = prompt_with_output_encoded[0].repeat(variants_tensor.size(0), 1)
        new_prompt_with_output_encoded[..., adv_indices] = variants_tensor

        return SimpleNamespace(
            new_prompt_with_output_encoded = new_prompt_with_output_encoded
        )
