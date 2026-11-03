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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from wsp.disguise import HEADS, LAYERS, disguise
from wsp.network import TokenShuffle, real_network
from wsp.reading import HINTED_SEED, head_scores, positive_heads, query_key_dot, stated_dot_products

ROOT = Path(__file__).parent.parent
ARMS = {"Alone, 2 runs": ("read-1", "read-2"), "With a formula and layer hint, 1 run": ("hinted",)}
SHOWN_LAYERS = range(4, 10)
INDUCTION_HEADS = 5
BLUE, RED, GREY, INK, MUTED = "#2a78d6", "#c8352b", "#e6e5e0", "#0b0b0b", "#52514e"
GAP = 0.06
FIGURE_INCHES, DPI = (10.8, 6.75), 200
LABEL_POINTS, TITLE_POINTS, RESULT_POINTS, STAT_POINTS = 20, 23, 26, 44
MARK_WIDTH = 4
TITLE = "Can GPT-6 Astra find induction heads in scrambled GPT-2 weights?"
EXAMPLE_INTRO = "An induction head looks back at what came next last time:"
EXAMPLE_WORDS = ("Harry", "Potter", "went", "home,", "and", "Harry")
NOTE = "Layers 4 to 9 of 12. A small 2 means picked in both runs."
ARITHMETIC_TITLE = "GPT-6 Astra typed this math on GPT-2's weights with no tool calls"
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


def _draw_cross(axis, x: float, y: float, half: float, width: float = MARK_WIDTH) -> None:
    for sign in (1, -1):
        axis.plot([x - half, x + half], [y - sign * half, y + sign * half], color=RED, linewidth=width, solid_capstyle="round", zorder=3)


def _draw_check(axis, x: float, y: float, half: float, down: int = 1) -> None:
    """Draws a white check mark, with `down` set to -1 on axes whose y grows upward."""
    xs, ys = [x - half, x - half * 0.2, x + half], [y, y + down * half * 0.75, y - down * half * 0.8]
    axis.plot(xs, ys, color="white", linewidth=MARK_WIDTH - 1, solid_capstyle="round", solid_joinstyle="round", zorder=3)


def _draw_grid(axis, scores: np.ndarray, induction_heads: set, picks: list[tuple[int, int]]) -> None:
    position_in_layer = np.argsort(np.argsort(-scores, axis=1), axis=1)
    if any(layer not in SHOWN_LAYERS for layer, _ in picks) or any(layer not in SHOWN_LAYERS for layer, _ in induction_heads):
        raise ValueError("a pick or an induction head falls outside the layers shown")
    for layer in SHOWN_LAYERS:
        axis.text(-0.35, layer + 0.5, str(layer), fontsize=LABEL_POINTS, ha="right", va="center", color=MUTED)
        for head in range(HEADS):
            colour = BLUE if (layer, head) in induction_heads else GREY
            left = position_in_layer[layer, head]
            axis.add_patch(Rectangle((left + GAP / 2, layer + GAP / 2), 1 - GAP, 1 - GAP, facecolor=colour, linewidth=0))
    for (layer, head), count in Counter(picks).items():
        left, middle = position_in_layer[layer, head], layer + 0.5
        if (layer, head) in induction_heads:
            _draw_check(axis, left + 0.5, middle, 0.27)
        elif count == 1:
            _draw_cross(axis, left + 0.5, middle, 0.22)
        else:
            _draw_cross(axis, left + 0.42, middle + 0.04, 0.2)
            axis.text(left + 0.86, middle - 0.2, str(count), fontsize=LABEL_POINTS - 6, fontweight="bold", ha="center", va="center", color=RED, zorder=4)
    axis.set(xlim=(0, HEADS), ylim=(SHOWN_LAYERS[-1] + 1, SHOWN_LAYERS[0]), aspect="equal")
    axis.axis("off")


def _draw_example(figure, height: float) -> None:
    """Draws a repeated word with an arrow from its second use back to the word that followed its first."""
    renderer = figure.canvas.get_renderer()
    figure.text(0.5, height + 0.075, EXAMPLE_INTRO, fontsize=LABEL_POINTS, ha="center", va="center", color=MUTED)
    placed, left = [], 0.0
    for index, word in enumerate(EXAMPLE_WORDS):
        colour = BLUE if index == 1 else INK
        weight = "bold" if index in (0, 1, 5) else "normal"
        text = figure.text(left, height, word, fontsize=RESULT_POINTS - 2, va="center", color=colour, fontweight=weight)
        box = text.get_window_extent(renderer).transformed(figure.transFigure.inverted())
        placed.append((text, box.width))
        left += box.width + 0.012
    shift = 0.5 - (left - 0.012) / 2
    centres = []
    for text, width in placed:
        x = text.get_position()[0] + shift
        text.set_x(x)
        centres.append(x + width / 2)
    figure.add_artist(FancyArrowPatch(
        (centres[5], height - 0.028), (centres[1], height - 0.028), transform=figure.transFigure,
        connectionstyle="arc3,rad=-0.22", arrowstyle="-|>", mutation_scale=20, linewidth=2, color=BLUE,
    ))


def _draw_legend(figure) -> None:
    height = FIGURE_INCHES[1] * 0.075
    axis = figure.add_axes([0, 0.075, 1, 0.075])
    axis.set(xlim=(0, FIGURE_INCHES[0]), ylim=(0, height), aspect="equal")
    axis.axis("off")
    middle, size = height / 2, 0.28
    for left, colour, text in ((0.75, GREY, "Other head"), (2.75, BLUE, "Induction head")):
        axis.add_patch(Rectangle((left, middle - size / 2), size, size, facecolor=colour, linewidth=0))
        axis.text(left + size + 0.12, middle, text, fontsize=LABEL_POINTS, va="center", color=INK)
    axis.add_patch(Rectangle((5.55, middle - size / 2), size, size, facecolor=BLUE, linewidth=0))
    _draw_check(axis, 5.55 + size / 2, middle, 0.09, down=-1)
    axis.text(5.95, middle, "Astra right", fontsize=LABEL_POINTS, va="center", color=INK)
    _draw_cross(axis, 7.9, middle, 0.09)
    axis.text(8.12, middle, "Astra wrong", fontsize=LABEL_POINTS, va="center", color=INK)


def _heads_figure(shuffle: TokenShuffle) -> None:
    real = real_network(shuffle).double()
    real.set_attn_implementation("eager")
    scores = head_scores(real)
    positives = positive_heads(scores)
    # The scores drop from 0.83 to 0.51 after fifth, and only those five are induction heads.
    order = np.argsort(-scores.ravel())[:INDUCTION_HEADS]
    induction_heads = {(int(i) // HEADS, int(i) % HEADS) for i in order}

    figure = plt.figure(figsize=FIGURE_INCHES, facecolor="white")
    figure.text(0.5, 0.955, TITLE, fontsize=TITLE_POINTS, ha="center", va="center", color=INK)
    _draw_example(figure, 0.8)
    panels = {}
    for index, (title, arms) in enumerate(ARMS.items()):
        results = [_run_result(arm) for arm in arms]
        picks = [pick for result in results for pick in _picks_in_original_order(result, shuffle)]
        hits = sum(pick in positives for pick in picks)
        if hits != sum(result["hits"] for result in results):
            raise ValueError(f"{title}: {hits} hits on the shared grid, stored {[result['hits'] for result in results]}")
        induction_hits = sum(pick in induction_heads for pick in picks)
        panels[title] = {
            "runs": list(arms), "picks": picks, "hits_top_10": hits, "hits_induction": induction_hits, "distinct_picks": len(set(picks)),
        }
        axis = figure.add_axes([0.065 + index * 0.485, 0.195, 0.42, 0.42])
        _draw_grid(axis, scores, induction_heads, picks)
        centre = HEADS / 2
        top = SHOWN_LAYERS[0]
        axis.text(centre, top - 1.5, title, fontsize=LABEL_POINTS, ha="center", color=INK)
        axis.text(centre, top - 0.35, f"{induction_hits} of {len(picks)} picks right", fontsize=RESULT_POINTS, fontweight="bold", ha="center", color=INK)
        axis.text(-0.35, top - 0.35, "Layer", fontsize=LABEL_POINTS - 3, ha="center", color=MUTED)
        bottom = SHOWN_LAYERS[-1] + 1.55
        axis.annotate("", xy=(0, bottom), xytext=(0.95, bottom), arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.5, "mutation_scale": 14, "shrinkA": 0, "shrinkB": 0}, annotation_clip=False)
        axis.text(1.1, bottom, "higher score", fontsize=LABEL_POINTS - 3, va="center", color=MUTED)
    _draw_legend(figure)
    figure.text(0.5, 0.045, NOTE, fontsize=LABEL_POINTS - 2, ha="center", va="center", color=MUTED)

    output = ROOT / "figures" / "reading_heads_v8.png"
    figure.savefig(output, dpi=DPI)
    data = {"positives": sorted(positives), "scores": scores.tolist(), "panels": panels}
    output.with_suffix(".json").write_text(json.dumps(data, indent=1))
    print(output, {title: {key: panel[key] for key in ("hits_top_10", "hits_induction")} for title, panel in panels.items()})


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
    matched = np.round(in_code, 2) == np.round(typed, 2)
    matches = int(matched.sum())
    largest_error = np.abs(typed - in_code).max()

    figure = plt.figure(figsize=FIGURE_INCHES, facecolor="white")
    axis = figure.add_axes([0.1, 0.14, 0.44, 0.72])
    axis.plot(AXIS_LIMITS, AXIS_LIMITS, color=GREY, linewidth=2.5, zorder=1)
    axis.text(1.03, 0.82, "exact match", rotation=45, rotation_mode="anchor", fontsize=LABEL_POINTS - 3, ha="center", color=MUTED)
    axis.scatter(in_code[matched], typed[matched], s=70, color=BLUE, edgecolor="white", linewidth=0.8, zorder=2)
    axis.scatter(in_code[~matched], typed[~matched], s=80, color=RED, edgecolor="white", linewidth=0.8, zorder=3)
    axis.set(xlim=AXIS_LIMITS, ylim=AXIS_LIMITS, xticks=AXIS_TICKS, yticks=AXIS_TICKS, aspect="equal")
    axis.set_xlabel("Computed in code", fontsize=LABEL_POINTS, color=INK)
    axis.set_ylabel("Typed by Astra", fontsize=LABEL_POINTS, color=INK)
    axis.tick_params(labelsize=LABEL_POINTS, colors=INK, length=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GREY)
    figure.text(0.5, 0.94, ARITHMETIC_TITLE, fontsize=TITLE_POINTS, ha="center", va="center", color=INK)
    stats = (
        (0.66, BLUE, f"{matches} of {len(keys)}", STAT_POINTS, "match code to the two\ndecimals it typed"),
        (0.38, RED, f"{len(keys) - matches} miss", STAT_POINTS - 12, f"by at most {largest_error:.3f}"),
    )
    for height, colour, value, size, text in stats:
        figure.add_artist(Line2D([0.595], [height + 0.03], transform=figure.transFigure, marker="o", markersize=11, color=colour, linewidth=0))
        figure.text(0.62, height, value, fontsize=size, fontweight="bold", color=INK)
        figure.text(0.62, height - 0.015, text, fontsize=LABEL_POINTS, color=INK, va="top", linespacing=1.25)
    figure.text(0.62, 0.2, "Each dot is one head:\na sum of 64 products.", fontsize=LABEL_POINTS, color=MUTED, va="center", linespacing=1.25)
    biggest = int(np.argmax(in_code))
    axis.annotate(
        f"typed {typed[biggest]:.2f}\ncode {in_code[biggest]:.2f}", (in_code[biggest], typed[biggest]), xytext=(0.42, 1.2),
        fontsize=LABEL_POINTS - 2, color=INK, va="center", ha="center", arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1.2, "shrinkB": 6},
    )

    output = ROOT / "figures" / "arithmetic_v5.png"
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
