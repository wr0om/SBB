import torch
import functools
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Prompt template for Vicuna v1.3 (weights v1.3) from:
# https://raw.githubusercontent.com/lm-sys/FastChat/main/docs/vicuna_weights_version.md#prompt-template
VICUNA_CHAT_TEMPLATE = """A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the user's questions.
USER: {instruction}
ASSISTANT: """

def format_instruction_vicuna_chat(
    instruction: str,
    output: str = None,
    include_trailing_whitespace: bool = True
) -> str:
    formatted = VICUNA_CHAT_TEMPLATE.format(instruction=instruction)
    if not include_trailing_whitespace:
        formatted = formatted.rstrip()
    if output is not None:
        formatted += output
    return formatted


def tokenize_instructions_vicuna_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace: bool = True
):
    if outputs is not None:
        prompts = [
            format_instruction_vicuna_chat(instr, out, include_trailing_whitespace)
            for instr, out in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_vicuna_chat(instr, include_trailing_whitespace=include_trailing_whitespace)
            for instr in instructions
        ]
    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


def orthogonalize_vicuna_weights(
    model,
    direction: Float[Tensor, "d_model"]
):
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


def act_add_vicuna_weights(
    model,
    direction: Float[Tensor, "d_model"],
    coeff: float,
    layer: int
):
    dtype = model.model.layers[layer - 1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer - 1].mlp.down_proj.weight.device
    bias = (coeff * direction).to(dtype=dtype, device=device)
    model.model.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)


class VicunaModel(ModelBase):

    def _load_model(self, model_path: str, dtype=torch.float16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_vicuna_chat,
            tokenizer=self.tokenizer,
            include_trailing_whitespace=True
        )

    def _get_eoi_toks(self):
        # Tail tokens after instruction: "\nASSISTANT: "
        return self.tokenizer.encode(
            VICUNA_CHAT_TEMPLATE.split("{instruction}")[-1],
            add_special_tokens=False
        )

    def _get_refusal_toks(self):
        # Using default refusal token ('I') from LLaMA vocabulary
        return [self.tokenizer.encode("I", add_special_tokens=False)[0]]

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([
            block.self_attn for block in self.model.model.layers
        ])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([
            block.mlp for block in self.model.model.layers
        ])

    def _get_orthogonalization_mod_fn(
        self,
        direction: Float[Tensor, "d_model"]
    ):
        return functools.partial(
            orthogonalize_vicuna_weights,
            direction=direction
        )

    def _get_act_add_mod_fn(
        self,
        direction: Float[Tensor, "d_model"],
        coeff: float,
        layer: int
    ):
        return functools.partial(
            act_add_vicuna_weights,
            direction=direction,
            coeff=coeff,
            layer=layer
        )
