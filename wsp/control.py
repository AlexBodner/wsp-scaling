"""The positive control: scores a Breakout policy as the post's prompt describes it."""

import sys
from pathlib import Path

import numpy as np
import torch

CONTROL_DATA = Path(__file__).parent.parent / "data" / "control"
_LOCAL_PACKAGES = Path(__file__).parent.parent / ".deps"
HIDDEN_SIZE = 32
INPUT_SIZE = 600
ACTIONS = 6
MAX_STEPS = 10000
BEST_TRAINED_SCORE = 25.3
# Their channel layout: player 0, danger 1, reward 2, projectile 3, trail 4, gauge 5.
_CHANNELS = {"paddle": 0, "brick": 2, "ball": 3, "trail": 4}


def zero_policy() -> dict[str, torch.Tensor]:
    """Returns the prompt's tensors with every weight at zero."""
    return {
        "W1": torch.zeros(INPUT_SIZE, HIDDEN_SIZE),
        "b1": torch.zeros(HIDDEN_SIZE),
        "W2": torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE),
        "b2": torch.zeros(HIDDEN_SIZE),
        "W3": torch.zeros(HIDDEN_SIZE, ACTIONS),
        "b3": torch.zeros(ACTIONS),
    }


def score_breakout(tensors: dict[str, torch.Tensor], episodes: int, seed: int) -> dict:
    """Returns the mean score per episode, with actions sampled and sticky actions at 0.1."""
    sys.path.insert(0, str(_LOCAL_PACKAGES))
    from minatar import Environment

    weights = {name: tensor.numpy() for name, tensor in tensors.items()}
    rng = np.random.default_rng(seed)
    environment = Environment("breakout", sticky_action_prob=0.1)
    environment.seed(seed)
    layout = [(index, _CHANNELS[name]) for name, index in environment.env.channels.items()]
    scores, lengths = [], []
    for _ in range(episodes):
        environment.reset()
        total, steps, done = 0.0, 0, False
        while not done and steps < MAX_STEPS:
            state = environment.state()
            observation = np.zeros((10, 10, 6), dtype=np.float32)
            for source, target in layout:
                observation[:, :, target] = np.maximum(observation[:, :, target], state[:, :, source])
            hidden = np.tanh(observation.reshape(-1) @ weights["W1"] + weights["b1"])
            hidden = np.tanh(hidden @ weights["W2"] + weights["b2"])
            logits = hidden @ weights["W3"] + weights["b3"]
            probabilities = np.exp(logits - logits.max())
            action = rng.choice(ACTIONS, p=probabilities / probabilities.sum())
            reward, done = environment.act(int(action))
            total += reward
            steps += 1
        scores.append(total)
        lengths.append(steps)
    return {
        "mean": float(np.mean(scores)),
        "standard_error": float(np.std(scores, ddof=1) / np.sqrt(episodes)),
        "episodes": episodes,
        "mean_length": float(np.mean(lengths)),
        "passes": bool(np.mean(scores) - 1.96 * np.std(scores, ddof=1) / np.sqrt(episodes) > BEST_TRAINED_SCORE),
    }
