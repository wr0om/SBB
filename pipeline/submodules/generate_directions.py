import torch
import os

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase

def get_mean_activations_pre_hook(layer, cache: Float[Tensor, "pos layer d_model"], n_samples, positions: List[int]):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn

def get_mean_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    fwd_pre_hooks = [(block_modules[layer], get_mean_activations_pre_hook(layer=layer, cache=mean_activations, n_samples=n_samples, positions=positions)) for layer in range(n_layers)]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return mean_activations

# ADDED BY US
def get_activations_array(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], target_output=None, batch_size=32, positions=[-1]):
    """
    Get activations for multiple instructions at specified positions.

    Args:
        model: The language model.
        tokenizer: The tokenizer for the model.
        instructions: A list of instructions (strings) to process.
        tokenize_instructions_fn: Function to tokenize the instructions.
        block_modules: List of model block modules to hook into.
        batch_size: Batch size for processing instructions.
        positions: List of token positions to extract activations from.

    Returns:
        activations_array: Tensor of shape (n_samples, n_positions, n_layers, d_model).
    """
    torch.cuda.empty_cache()

    n_samples = len(instructions)
    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    # Initialize a tensor to store activations for all samples
    activations_array = torch.zeros((n_samples, n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    # Process instructions in batches
    for i in tqdm(range(0, n_samples, batch_size)):
        batch_instructions = instructions[i:i + batch_size]
        batch_size_actual = len(batch_instructions)

        # Temporary tensor to store activations for the current batch
        batch_activations = torch.zeros((batch_size_actual, n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

        # Define hooks for each layer
        fwd_pre_hooks = [
            (block_modules[layer], get_mean_activations_pre_hook(layer=layer, cache=batch_activations[j], n_samples=1, positions=positions))
            for j in range(batch_size_actual)
            for layer in range(n_layers)
        ]

        # Tokenize the batch of instructions
        if target_output:
            outputs = [target_output] * batch_size_actual
        else:
            outputs = None
        inputs = tokenize_instructions_fn(instructions=batch_instructions, outputs=outputs)

        # Run the model with hooks to capture activations
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

        # Store batch activations in the main array
        activations_array[i:i + batch_size_actual] = batch_activations

    return activations_array

def get_mean_diff(model, tokenizer, harmful_instructions, harmless_instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    mean_activations_harmful = get_mean_activations(model, tokenizer, harmful_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)
    mean_activations_harmless = get_mean_activations(model, tokenizer, harmless_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_harmful - mean_activations_harmless

    return mean_diff

def generate_directions(model_base: ModelBase, harmful_instructions, harmless_instructions, artifact_dir):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    mean_diffs = get_mean_diff(model_base.model, model_base.tokenizer, harmful_instructions, harmless_instructions, model_base.tokenize_instructions_fn, model_base.model_block_modules, positions=list(range(-len(model_base.eoi_toks), 0))) # TAKE THE FINAL SPECIAL TOKENS IN THE END

    assert mean_diffs.shape == (len(model_base.eoi_toks), model_base.model.config.num_hidden_layers, model_base.model.config.hidden_size)
    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")

    return mean_diffs