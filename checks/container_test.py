"""Tests whether Astra's written networks use GPT-2's size: reruns each reply with most layers deleted."""
import torch
from transformers import AutoConfig, GPT2LMHeadModel

from wsp.edits import apply_edits
from wsp.network import TokenShuffle, writable_tensors
from wsp.tasks import score_capitals, score_reverse, top_tokens

REPLIES = {
    "capitals": "runs/capitals__live__20261103T002438Z/reply.md",
    "reverse": "runs/reverse__live__20261103T004038Z/reply.md",
}


def built(reply_path, layers):
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = layers
    model = GPT2LMHeadModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    report = apply_edits(open(reply_path).read(), writable_tensors(model))
    return model, report


shuffle = TokenShuffle()
for layers in (12, 2, 1):
    model, report = built(REPLIES["capitals"], layers)
    score = score_capitals(model, shuffle)
    print(f"capitals layers={layers} weights={sum(p.numel() for p in model.parameters()):,} applied={report.applied}/{report.statements} visible={score['visible'].value:.4f} hidden={score['hidden'].value:.4f}")
for layers in (12, 2, 1):
    model, report = built(REPLIES["reverse"], layers)
    print(f"reverse  layers={layers} weights={sum(p.numel() for p in model.parameters()):,} applied={report.applied}/{report.statements} exact={score_reverse(model, shuffle)['exact'].value:.4f}")

model, _ = built(REPLIES["capitals"], 12)
print("nonzero share", sum(int((p != 0).sum()) for p in model.parameters()) / sum(p.numel() for p in model.parameters()))
for text in (
    "The president of France is",
    "The capital of France is",
    "The population of Japan is",
    "Tell me about Japan",
    "France",
    "The capital of Paris is",
    "The capital of France and Japan is",
):
    print(repr(text), "->", repr(shuffle.decode_token(top_tokens(model, [shuffle.encode(text)])[0])))
