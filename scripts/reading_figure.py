"""Draws the post figures for the reading test: where Astra's picks landed, and its typed arithmetic against code.

Each disguise reorders heads differently, so picks are mapped back to GPT-2's original heads to share one grid.
Heads are sorted by copying score within each layer, so distance along a row means something.
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from wsp.disguise import HEADS, LAYERS, disguise
from wsp.network import TokenShuffle, real_network
from wsp.reading import HINTED_SEED, head_scores, positive_heads, query_key_dot, stated_dot_products

ROOT = Path(__file__).parent.parent
ARMS = {"On its own": ("read-1", "read-2"), "Given the formula and where to look": ("hinted",)}
HINTED_ARM = "Given the formula and where to look"
LAYERS_TOLD_TO_SKIP = 4
BLUE, GREY, INK, MUTED = "#2a78d6", "#e6e5e0", "#0b0b0b", "#52514e"
GAP = 0.06
FIGURE_INCHES, DPI = (10.8, 6.75), 200
LABEL_POINTS, TITLE_POINTS, RESULT_POINTS, STAT_POINTS = 20, 23, 24, 42
RING_WIDTH = 2.4
TITLE = "Can GPT-6 Astra find the heads in GPT-2 that look back at repeated text?"
SUBTITLE = "No tool calls in our logs. Rows are layers, heads sorted by that score."
ARITHMETIC_TITLE = "GPT-6 Astra's typed results against code, no tool calls"
# The row and the turns where the hinted read typed a dot product for every head in layers 4 to 11.
HINTED_TABLE_ROW, HINTED_TABLE_TURNS, HINTED_TABLE_VALUES = 137, (1, 2), 96
AXIS_LIMITS = (-0.75, 1.35)
AXIS_TICKS = (-0.5, 0, 0.5, 1)


def _run_folder(arm: str) -> Path:
    folders = sorted((ROOT / "runs").glob(f"reading-{arm}__live__*"))
    if len(folders) != 1:
        raise ValueError(f"expected one live run for {arm}, found {len(folders)}")
    return folders[0]


def _run_result(arm: str) -> dict:
    return json.loads((_run_folder(arm) / "result.json").read_text())


def _picks_in_original_order(result: dict, shuffle: TokenShuffle) -> list[tuple[int, int]]:
    original_head = disguise(result["seed"], shuffle).original_head
    return [(layer, int(original_head[layer][head])) for layer, head in result["answer"]]


def _draw_ring(axis, x: float, y: float, radius: float) -> None:
    axis.add_patch(Circle((x, y), radius, facecolor="white", edgecolor=INK, linewidth=RING_WIDTH, zorder=3))


def _draw_grid(axis, scores: np.ndarray, positives: set, picks: list[tuple[int, int]]) -> None:
    position_in_layer = np.argsort(np.argsort(-scores, axis=1), axis=1)
    for layer in range(LAYERS):
        for head in range(HEADS):
            colour = BLUE if (layer, head) in positives else GREY
            left = position_in_layer[layer, head]
            axis.add_patch(Rectangle((left + GAP / 2, layer + GAP / 2), 1 - GAP, 1 - GAP, facecolor=colour, linewidth=0))
    for (layer, head), count in Counter(picks).items():
        left = position_in_layer[layer, head]
        if count == 1:
            _draw_ring(axis, left + 0.5, layer + 0.5, 0.3)
        else:
            _draw_ring(axis, left + 0.27, layer + 0.5, 0.19)
            _draw_ring(axis, left + 0.73, layer + 0.5, 0.19)
    axis.set(xlim=(0, HEADS), ylim=(LAYERS, 0), aspect="equal")
    axis.axis("off")


def _draw_legend(figure) -> None:
    height = FIGURE_INCHES[1] * 0.09
    axis = figure.add_axes([0, 0, 1, 0.09])
    axis.set(xlim=(0, FIGURE_INCHES[0]), ylim=(0, height))
    axis.axis("off")
    middle, size = height / 2, 0.3
    for left, colour, text in ((0.55, GREY, "Head"), (1.85, BLUE, "Head that looks back")):
        axis.add_patch(Rectangle((left, middle - size / 2), size, size, facecolor=colour, linewidth=0))
        axis.text(left + size + 0.12, middle, text, fontsize=LABEL_POINTS, va="center", color=INK)
    _draw_ring(axis, 5.7, middle, 0.13)
    axis.text(5.93, middle, "Astra's pick", fontsize=LABEL_POINTS, va="center", color=INK)
    _draw_ring(axis, 7.95, middle, 0.1)
    _draw_ring(axis, 8.2, middle, 0.1)
    axis.text(8.4, middle, "Picked twice", fontsize=LABEL_POINTS, va="center", color=INK)


def _heads_figure(shuffle: TokenShuffle) -> None:
    real = real_network(shuffle).double()
    real.set_attn_implementation("eager")
    scores = head_scores(real)
    positives = positive_heads(scores)

    figure = plt.figure(figsize=FIGURE_INCHES, facecolor="white")
    figure.text(0.5, 0.95, TITLE, fontsize=TITLE_POINTS, ha="center", va="center", color=INK)
    figure.text(0.5, 0.895, SUBTITLE, fontsize=LABEL_POINTS, ha="center", va="center", color=MUTED)
    panels = {}
    for index, (title, arms) in enumerate(ARMS.items()):
        results = [_run_result(arm) for arm in arms]
        picks = [pick for result in results for pick in _picks_in_original_order(result, shuffle)]
        hits = sum(pick in positives for pick in picks)
        if hits != sum(result["hits"] for result in results):
            raise ValueError(f"{title}: {hits} hits on the shared grid, stored {[result['hits'] for result in results]}")
        panels[title] = {"runs": list(arms), "picks": picks, "hits": hits, "distinct_picks": len(set(picks))}
        axis = figure.add_axes([0.05 + index * 0.49, 0.1, 0.42, 0.62])
        _draw_grid(axis, scores, positives, picks)
        centre = HEADS / 2
        axis.text(centre, -1.75, title, fontsize=LABEL_POINTS, ha="center", color=INK)
        axis.text(centre, -0.45, f"{hits} of {len(picks)} picks right", fontsize=RESULT_POINTS, fontweight="bold", ha="center", color=INK)
        if title == HINTED_ARM:
            axis.text(centre, LAYERS_TOLD_TO_SKIP / 2, "told to skip these layers", fontsize=LABEL_POINTS, ha="center", va="center", color=MUTED)
    _draw_legend(figure)

    output = ROOT / "figures" / "reading_heads_v3.png"
    figure.savefig(output, dpi=DPI)
    data = {"positives": sorted(positives), "scores": scores.tolist(), "panels": panels}
    output.with_suffix(".json").write_text(json.dumps(data, indent=1))
    print(output, {title: {key: panel[key] for key in ("hits", "distinct_picks")} for title, panel in panels.items()})


def _arithmetic_figure(shuffle: TokenShuffle) -> None:
    lines = (_run_folder("hinted") / "transcript.jsonl").read_text().splitlines()
    turns = [json.loads(line) for line in lines]
    stated = stated_dot_products([turns[index]["reply"] for index in HINTED_TABLE_TURNS])
    if len(stated) != HINTED_TABLE_VALUES:
        raise ValueError(f"expected {HINTED_TABLE_VALUES} stated values, parsed {len(stated)}")
    disguised = disguise(HINTED_SEED, shuffle)
    keys = sorted(stated)
    typed = np.array([stated[key] for key in keys])
    in_code = np.array([query_key_dot(disguised, *key, HINTED_TABLE_ROW) for key in keys])
    correlation = np.corrcoef(typed, in_code)[0, 1]
    matches = int((np.round(in_code, 2) == np.round(typed, 2)).sum())
    largest_error = np.abs(typed - in_code).max()

    figure = plt.figure(figsize=FIGURE_INCHES, facecolor="white")
    axis = figure.add_axes([0.1, 0.14, 0.44, 0.72])
    axis.plot(AXIS_LIMITS, AXIS_LIMITS, color=GREY, linewidth=2.5, zorder=1)
    axis.scatter(in_code, typed, s=90, color=BLUE, edgecolor="white", linewidth=1.2, zorder=2)
    axis.set(xlim=AXIS_LIMITS, ylim=AXIS_LIMITS, xticks=AXIS_TICKS, yticks=AXIS_TICKS, aspect="equal")
    axis.set_xlabel("Computed in code", fontsize=LABEL_POINTS, color=INK)
    axis.set_ylabel("Typed by Astra", fontsize=LABEL_POINTS, color=INK)
    axis.tick_params(labelsize=LABEL_POINTS, colors=INK, length=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GREY)
    figure.text(0.5, 0.94, ARITHMETIC_TITLE, fontsize=TITLE_POINTS, ha="center", va="center", color=INK)
    stats = ((0.68, f"{matches} of {len(keys)}", "match code to the two\ndecimals it typed"), (0.39, f"{correlation:.3f}", "correlation with code"))
    for height, value, text in stats:
        figure.text(0.62, height, value, fontsize=STAT_POINTS, fontweight="bold", color=INK)
        figure.text(0.62, height - 0.015, text, fontsize=LABEL_POINTS, color=INK, va="top", linespacing=1.25)
    figure.text(0.62, 0.2, f"Each dot is one head, a sum\nof 64 products. Largest\nerror {largest_error:.3f}.", fontsize=LABEL_POINTS, color=MUTED, linespacing=1.25, va="center")

    output = ROOT / "figures" / "arithmetic_v3.png"
    figure.savefig(output, dpi=DPI)
    pairs = [{"layer": key[0], "head": key[1], "typed": float(value), "in_code": float(code)} for key, value, code in zip(keys, typed, in_code)]
    summary = {"row": HINTED_TABLE_ROW, "correlation": correlation, "matches_two_decimals": matches, "largest_error": largest_error}
    output.with_suffix(".json").write_text(json.dumps({**summary, "pairs": pairs}, indent=1))
    print(output, summary)


def main() -> None:
    plt.rcParams["font.family"] = "Helvetica"
    shuffle = TokenShuffle()
    _arithmetic_figure(shuffle)
    _heads_figure(shuffle)


if __name__ == "__main__":
    main()
