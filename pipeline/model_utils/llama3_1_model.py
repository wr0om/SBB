import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Llama 3.1 default system prompt based on Meta's spec
# Cutting knowledge date and tool-calling guidance
LLAMA3_1_DEFAULT_SYSTEM_PROMPT = """Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

You are a helpful assistant
"""

# Chat templates
LLAMA3_1_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

LLAMA3_1_CHAT_TEMPLATE_WITH_SYSTEM = """<|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# Refusal tokens (e.g., start of "I").
LLAMA3_1_REFUSAL_TOKS = [40]

def format_instruction_llama3_1(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    if system is not None:
        formatted = LLAMA3_1_CHAT_TEMPLATE_WITH_SYSTEM.format(
            system_prompt=system,
            instruction=instruction
        )
    else:
        formatted = LLAMA3_1_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted = formatted.rstrip()

    if output is not None:
        formatted += output

    return formatted

def tokenize_instructions_llama3_1(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    if outputs is not None:
        prompts = [
            format_instruction_llama3_1(instruction=instr, output=out, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instr, out in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_llama3_1(instruction=instr, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instr in instructions
        ]

    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt"
    )

def orthogonalize_llama3_1_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)
    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_llama3_1_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer-1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer-1].mlp.down_proj.weight.device
    bias = (coeff * direction).to(dtype=dtype, device=device)
    model.model.layers[layer-1].mlp.down_proj.bias = torch.nn.Parameter(bias)

class Llama3_1Model(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_llama3_1,
            tokenizer=self.tokenizer,
            system=LLAMA3_1_DEFAULT_SYSTEM_PROMPT,
            include_trailing_whitespace=True
        )

    def _get_eoi_toks(self):
        # tokens for end-of-instruction marker
        return self.tokenizer.encode(LLAMA3_1_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return LLAMA3_1_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block.self_attn for block in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.mlp for block in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_llama3_1_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_llama3_1_weights, direction=direction, coeff=coeff, layer=layer)
