"""GPT-2 small under neutral tensor names, and its tokenizer with shuffled token IDs."""

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel

MODEL_NAME = "gpt2"
SHUFFLE_SEED = 20261102


def _neutral_names(layers: int) -> dict[str, str]:
    names = {
        "E": "transformer.wte.weight",
        "P": "transformer.wpe.weight",
        "F.scale": "transformer.ln_f.weight",
        "F.shift": "transformer.ln_f.bias",
    }
    for layer in range(layers):
        source = f"transformer.h.{layer}"
        names |= {
            f"L{layer}.norm1.scale": f"{source}.ln_1.weight",
            f"L{layer}.norm1.shift": f"{source}.ln_1.bias",
            f"L{layer}.qkv.W": f"{source}.attn.c_attn.weight",
            f"L{layer}.qkv.b": f"{source}.attn.c_attn.bias",
            f"L{layer}.out.W": f"{source}.attn.c_proj.weight",
            f"L{layer}.out.b": f"{source}.attn.c_proj.bias",
            f"L{layer}.norm2.scale": f"{source}.ln_2.weight",
            f"L{layer}.norm2.shift": f"{source}.ln_2.bias",
            f"L{layer}.up.W": f"{source}.mlp.c_fc.weight",
            f"L{layer}.up.b": f"{source}.mlp.c_fc.bias",
            f"L{layer}.down.W": f"{source}.mlp.c_proj.weight",
            f"L{layer}.down.b": f"{source}.mlp.c_proj.bias",
        }
    return names


class TokenShuffle:
    """The tokenizer with every token renumbered by one fixed random order."""

    def __init__(self, seed: int = SHUFFLE_SEED) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.to_shuffled = np.random.default_rng(seed).permutation(len(self.tokenizer))
        self.to_original = np.argsort(self.to_shuffled)

    def encode(self, text: str) -> list[int]:
        """Returns the shuffled token IDs of a text."""
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return self.to_shuffled[ids].tolist()

    def decode_token(self, shuffled_id: int) -> str:
        """Returns the text of one shuffled token ID."""
        return self.tokenizer.decode([int(self.to_original[shuffled_id])])


def zero_network() -> GPT2LMHeadModel:
    """Returns GPT-2 small with every weight at zero."""
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return model


def real_network(shuffle: TokenShuffle) -> GPT2LMHeadModel:
    """Returns trained GPT-2 small with its token table renumbered to the shuffled IDs."""
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).eval()
    with torch.no_grad():
        table = model.transformer.wte.weight
        table[torch.from_numpy(shuffle.to_shuffled)] = table.clone()
    return model


def writable_tensors(model: GPT2LMHeadModel) -> dict[str, torch.Tensor]:
    """Returns the model's parameters under the neutral names shown to the writer."""
    parameters = dict(model.named_parameters())
    return {name: parameters[source] for name, source in _neutral_names(model.config.n_layer).items()}
