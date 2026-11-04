"""Sets the bar for the read-then-write task: how far simple edits lower repeating, and what they cost on real text. Makes no Astra calls."""
import numpy as np

from wsp.disguise import HEAD_SIZE, HEADS, disguise
from wsp.network import TokenShuffle
from wsp.write_task import apply_edit, score_disguised

SEED = 1399
TOP_FIVE = [(5, 5), (7, 10), (6, 9), (5, 1), (7, 2)]
NEXT_FIVE = [(9, 6), (10, 1), (9, 9), (5, 0), (10, 7)]
RANDOM_SETS = 6


def zeroed_outputs(disguised, real_heads: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    """Returns edits that zero the given real heads' rows in the output matrix, found through the scramble's map."""
    edited: dict[str, np.ndarray] = {}
    for layer, head in real_heads:
        name = f"L{layer}.out.W"
        values = edited.get(name, disguised.tensors[name].copy())
        scrambled = int(np.where(disguised.original_head[layer] == head)[0][0])
        values[scrambled * HEAD_SIZE : (scrambled + 1) * HEAD_SIZE] = 0
        edited[name] = values
    return edited


def main() -> None:
    shuffle = TokenShuffle()
    disguised = disguise(SEED, shuffle)
    base = score_disguised(disguised, shuffle)
    print(f"unedited: repeat accuracy {base[0]:.3f}, text loss {base[1]:.3f}")
    for label, heads in (("the top 5 zeroed", TOP_FIVE), ("the next 5 zeroed", NEXT_FIVE), ("the top 10 zeroed", TOP_FIVE + NEXT_FIVE)):
        repeat, text = score_disguised(apply_edit(disguised, zeroed_outputs(disguised, heads)), shuffle)
        print(f"{label}: repeat accuracy {repeat:.3f}, text loss {text:.3f}, rise {text - base[1]:.3f}")
    rng = np.random.default_rng(SEED)
    others = [(layer, head) for layer in range(12) for head in range(HEADS) if (layer, head) not in TOP_FIVE + NEXT_FIVE]
    results = np.array([score_disguised(apply_edit(disguised, zeroed_outputs(disguised, [others[i] for i in rng.choice(len(others), 5, replace=False)])), shuffle) for _ in range(RANDOM_SETS)])
    print(f"{RANDOM_SETS} random sets of 5 zeroed: repeat accuracy mean {results[:, 0].mean():.3f}, lowest {results[:, 0].min():.3f}; text loss rise mean {results[:, 1].mean() - base[1]:.3f}, highest {results[:, 1].max() - base[1]:.3f}")


if __name__ == "__main__":
    main()
