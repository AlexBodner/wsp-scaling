"""Why LayerNorm makes hand-written weights harder: a one-hot identity network with and without it."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads, norm):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.n1 = nn.LayerNorm(d) if norm else nn.Identity()
        self.n2 = nn.LayerNorm(d) if norm else nn.Identity()

    def forward(self, x):
        B, T, D = x.shape
        chunks = self.qkv(self.n1(x)).chunk(3, -1)
        q, k, v = [c.view(B, T, self.heads, D // self.heads).transpose(1, 2) for c in chunks]
        w = (q @ k.transpose(-1, -2)) / (D // self.heads) ** 0.5
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        w = w.masked_fill(mask, -1e9).softmax(-1)
        x = x + self.proj((w @ v).transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.relu(self.fc1(self.n2(x))))


class Net(nn.Module):
    def __init__(self, vocab, context, layers, d, heads, norm):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(context, d)
        self.blocks = nn.ModuleList(Block(d, heads, norm) for _ in range(layers))
        self.final = nn.LayerNorm(d) if norm else nn.Identity()
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1]))
        for block in self.blocks:
            x = block(x)
        return self.head(self.final(x))


VOCAB, CONTEXT = 41, 64
ids = torch.randint(0, VOCAB, (2, 10))
for norm in (False, True):
    net = Net(VOCAB, CONTEXT, 2, 64, 4, norm)
    print("norm", norm, "weights", f"{sum(p.numel() for p in net.parameters()):,}")
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        print("  all zero, max logit", float(net(ids).abs().max()))
        index = torch.arange(VOCAB)
        net.tok.weight[index, index] = 1.0
        net.head.weight[index, index] = 1.0
        out = net(ids)
        print("  one-hot identity, signal", round(float(out.gather(-1, ids[..., None]).mean()), 3))
        if norm:
            for name, p in net.named_parameters():
                if any(key in name for key in ("n1", "n2", "final")) and name.endswith("weight"):
                    p.fill_(1.0)
            out = net(ids)
            print("  norm scales at 1, signal", round(float(out.gather(-1, ids[..., None]).mean()), 3))
