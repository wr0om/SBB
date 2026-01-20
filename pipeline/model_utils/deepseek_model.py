import torch
import functools
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List
from torch import Tensor
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Chat template aligned with official example:
DEEPSEEK_CHAT_TEMPLATE = "User: {instruction}\n\nAssistant:"
DEEPSEEK_REFUSAL_TOKS = []  # optional, fill if needed

def format_instruction_deepseek_chat(instruction: str, output: str = None):
    prompt = DEEPSEEK_CHAT_TEMPLATE.format(instruction=instruction)
    if output is not None:
        prompt += " " + output
    return prompt

def tokenize_instructions_deepseek_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None
):
    if outputs is not None:
        prompts = [
            format_instruction_deepseek_chat(inst, out)
            for inst, out in zip(instructions, outputs)
        ]
    else:
        prompts = [format_instruction_deepseek_chat(inst) for inst in instructions]
    return tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")

def orthogonalize_deepseek_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction
    )
    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.down_proj.weight.data.T, direction
        ).T

def act_add_deepseek_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    blk = model.model.layers[layer - 1]
    dtype = blk.mlp.down_proj.weight.dtype
    device = blk.mlp.down_proj.weight.device
    bias = (coeff * direction).to(dtype=dtype, device=device)
    blk.mlp.down_proj.bias = torch.nn.Parameter(bias)

class DeepSeek7BChatModel(ModelBase):
    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        # Load official generation config and set padding token
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_deepseek_chat,
            tokenizer=self.tokenizer
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode("Assistant:", add_special_tokens=False)

    def _get_refusal_toks(self):
        return DEEPSEEK_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            blk.self_attn for blk in self._get_model_block_modules()
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(
            blk.mlp for blk in self._get_model_block_modules()
        )

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_deepseek_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_deepseek_weights,
            direction=direction,
            coeff=coeff,
            layer=layer
        )
