"""Draws the post figures for the reading test: where Astra's picks landed, and its typed arithmetic against code.

Each disguise reorders heads differently, so picks are mapped back to GPT-2's original heads to share one grid.
Heads are sorted by copying score within each layer, so distance along a row means something.
"""

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH, disguise
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
EXAMPLE_INTRO = "An induction head finds what came next last time, and copies it:"
EXAMPLE_WORDS = ("road", "bird", "hand", "road", "bird")
EXAMPLE_ANSWER = 2
NOTE = "Layers 4 to 9 of 12. A small 2 means picked in both runs."
STORY_TITLE = "GPT-6 Astra hunts induction heads in scrambled GPT-2"
STORY_STEPS = (("Not shown any weights", ("blind",)), ("No tools, shown 20,000 values", ("read-1", "read-2")), ("With Python, all 12 blocks", ("python-1", "python-2")))
STORY_NO_WEIGHTS_RESULT = "Names the right layers"
STORY_NOTE = "No method given. Rows are layers 4 to 9 of 12, induction heads first. A small 2 means picked twice."
STORY_PYTHON_NOTE = "Both runs picked these same 5"
STORY_READ_NOTE = "Too few values to run this method"
STORY_POINTS = 16
METHOD_QUOTE = "an earlier head can write the preceding token's features into a position, and a later head can match those features against the current token."
METHOD_CODE = ("A, B = Q @ K.T, V @ O", "raw = As.reshape(144, -1) @ Bs.reshape(144, -1).T", "cos = raw / (stats[:, 0, None] * stats[None, :, 2])")
METHOD_CODE_SHOWN = ("A, B = Q @ K.T, V @ O", "raw = As.reshape(144, -1) @ Bs.reshape(144, -1).T", "cos = raw / (stats[:, 0, None] * stats[None, :, 2])")
ARITHMETIC_TITLE = "Given our formula, Astra gave 96 sums like this, no tool calls logged"
EXAMPLE_TERMS = 3
# The row and the turns where the hinted read typed a dot product for every head in layers 4 to 11.
HINTED_TABLE_ROW, HINTED_TABLE_TURNS, HINTED_TABLE_VALUES = 137, (1, 2), 96


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
        colour = BLUE if index == EXAMPLE_ANSWER else INK
        weight = "bold" if index in (EXAMPLE_ANSWER, len(EXAMPLE_WORDS) - 1) else "normal"
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
        (centres[-1], height - 0.028), (centres[EXAMPLE_ANSWER], height - 0.028), transform=figure.transFigure,
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
    _draw_example(figure, 0.775)
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
    figure.text(0.5, 0.945, ARITHMETIC_TITLE, fontsize=TITLE_POINTS, ha="center", va="center", color=INK)
    biggest = int(np.argmax(in_code))
    layer, head = keys[biggest]
    row = disguised.tensors[f"L{layer}.qkv.W"][HINTED_TABLE_ROW]
    query, key = row[head * HEAD_SIZE : (head + 1) * HEAD_SIZE], row[WIDTH + head * HEAD_SIZE : WIDTH + (head + 1) * HEAD_SIZE]
    terms = [f"({query[i]:.3g} \u00d7 {key[i]:.3g})".replace("-", "\u2212") for i in range(EXAMPLE_TERMS)]
    figure.text(0.5, 0.82, " + ".join(terms) + " + ...", fontsize=LABEL_POINTS + 2, ha="center", va="center", color=INK)
    figure.text(0.5, 0.74, f"64 products in all. Astra's answer: {typed[biggest]:.2f}. Code: {in_code[biggest]:.2f}.", fontsize=LABEL_POINTS, ha="center", va="center", color=MUTED)

    axis = figure.add_axes([0.075, 0.13, 0.5, 0.5])
    first_layer = min(key[0] for key in keys)
    for (pair_layer, pair_head), is_match in zip(keys, matched):
        axis.add_patch(Rectangle((pair_head + GAP / 2, pair_layer - first_layer + GAP / 2), 1 - GAP, 1 - GAP, facecolor=BLUE if is_match else RED, linewidth=0))
    rows = max(key[0] for key in keys) - first_layer + 1
    axis.set(xlim=(0, HEADS), ylim=(rows, 0), aspect="equal")
    axis.axis("off")
    axis.text(HEADS / 2, -0.35, "All 96 sums, one square each", fontsize=LABEL_POINTS, ha="center", color=INK)

    stats = (
        (0.46, BLUE, f"{matches} of {len(keys)}", STAT_POINTS, "match code to the two\ndecimals it gave"),
        (0.22, RED, f"{len(keys) - matches} miss", STAT_POINTS - 12, f"by at most {largest_error:.3f}"),
    )
    for height, colour, value, size, text in stats:
        figure.add_artist(Line2D([0.635], [height + 0.03], transform=figure.transFigure, marker="s", markersize=13, color=colour, linewidth=0))
        figure.text(0.66, height, value, fontsize=size, fontweight="bold", color=INK)
        figure.text(0.66, height - 0.015, text, fontsize=LABEL_POINTS, color=INK, va="top", linespacing=1.25)

    output = ROOT / "figures" / "arithmetic_v9.png"
    figure.savefig(output, dpi=DPI)
    pairs = [{"layer": key[0], "head": key[1], "typed": float(value), "in_code": float(code)} for key, value, code in zip(keys, typed, in_code)]
    summary = {"row": HINTED_TABLE_ROW, "correlation": correlation, "matches_two_decimals": matches, "largest_error": largest_error}
    output.with_suffix(".json").write_text(json.dumps({**summary, "pairs": pairs}, indent=1))
    print(output, summary)


def _story_grid(axis, scores: np.ndarray, induction_heads: set, picks: list[tuple[int, int]], layer_counts: Counter | None) -> None:
    """Draws one step of the story: marks on heads, or for the no-weights call, how many picks each layer got."""
    _draw_grid(axis, scores, induction_heads, picks)
    for layer, count in (layer_counts or {}).items():
        word = "pick" if count == 1 else "picks"
        axis.text(HEADS - 0.25, layer + 0.5, f"{count} {word}", fontsize=STORY_POINTS, fontweight="bold", ha="right", va="center", color=INK, zorder=4)


def _story_figure(shuffle: TokenShuffle) -> None:
    real = real_network(shuffle).double()
    real.set_attn_implementation("eager")
    scores = head_scores(real)
    order = np.argsort(-scores.ravel())[:INDUCTION_HEADS]
    induction_heads = {(int(i) // HEADS, int(i) % HEADS) for i in order}
    blind = _run_result("blind")
    layer_counts = Counter(layer for layer, _ in blind["answer"])
    if dict(layer_counts) != dict(Counter(layer for layer, _ in induction_heads)):
        raise ValueError(f"the no-weights layers {layer_counts} no longer match the induction heads")

    figure = plt.figure(figsize=FIGURE_INCHES, facecolor="white")
    figure.text(0.5, 0.955, STORY_TITLE, fontsize=TITLE_POINTS, ha="center", va="center", color=INK)
    _draw_example(figure, 0.775)
    results = {}
    for index, (heading, arms) in enumerate(STORY_STEPS):
        picks = [pick for arm in arms for pick in _picks_in_original_order(_run_result(arm), shuffle)]
        hits = sum(pick in induction_heads for pick in picks)
        result = STORY_NO_WEIGHTS_RESULT if not index else f"{hits // len(arms)} of {INDUCTION_HEADS}, twice"
        results[heading] = {"runs": list(arms), "picks": picks, "hits_induction": hits}
        axis = figure.add_axes([0.042 + index * 0.325, 0.24, 0.292, 0.28])
        _story_grid(axis, scores, induction_heads, [] if not index else picks, layer_counts if not index else None)
        axis.text(HEADS / 2, 2.15, heading, fontsize=STORY_POINTS + 1, ha="center", color=INK)
        axis.text(HEADS / 2, 3.4, result, fontsize=RESULT_POINTS - 3, fontweight="bold", ha="center", color=INK)
        if index:
            note = STORY_PYTHON_NOTE if arms[0].startswith("python") else STORY_READ_NOTE
            axis.text(HEADS / 2, SHOWN_LAYERS[-1] + 1.7, note, fontsize=STORY_POINTS, ha="center", va="center", color=MUTED)
    _draw_legend(figure)
    figure.text(0.5, 0.045, STORY_NOTE, fontsize=LABEL_POINTS - 3, ha="center", va="center", color=MUTED)
    output = ROOT / "figures" / "reading_story_v7.png"
    figure.savefig(output, dpi=DPI)
    output.with_suffix(".json").write_text(json.dumps({"blind_layers": sorted(layer_counts.elements()), "steps": results}, indent=1))
    print(output, {heading: value["hits_induction"] for heading, value in results.items()})


def _method_figure() -> None:
    """Shows, verbatim from run 2, what Astra said, its key lines of code and what they printed."""
    turns = [json.loads(line) for line in (_run_folder("python-2") / "transcript.jsonl").read_text().splitlines()]
    words = turns[0]["reply"].split("```python")[0]
    if METHOD_QUOTE not in words.replace("\u2019", "'"):
        raise ValueError("the quote is not in the run's reply")
    for line in METHOD_CODE:
        if line not in turns[0]["code"]:
            raise ValueError(f"{line!r} is not in the run's code")
    answer = [f"{layer}.{head}" for layer, head in _run_result("python-2")["answer"]]
    cosines = turns[0]["output"].split("TOP 35 CIRCUITS: COSINE")[1].splitlines()[1:8]
    table = [re.match(r"\s*(\S+) <-\s+(\S+): raw=\s*\S+ cos=(\S+)", line).groups() for line in cosines]
    figure = plt.figure(figsize=FIGURE_INCHES, facecolor="white")
    figure.text(0.5, 0.945, "How GPT-6 Astra found them (Python run 2, first code)", fontsize=TITLE_POINTS, ha="center", va="center", color=INK)
    figure.text(0.05, 0.855, "What it wrote before any code", fontsize=LABEL_POINTS - 2, color=MUTED, va="center")
    figure.text(0.05, 0.76, "\"...an earlier head can write the preceding token's\nfeatures into a position, and a later head can match\nthose features against the current token.\"", fontsize=LABEL_POINTS, color=INK, va="center", linespacing=1.3)
    figure.text(0.05, 0.635, "Its key lines", fontsize=LABEL_POINTS - 2, color=MUTED, va="center")
    figure.text(0.05, 0.55, "\n".join(METHOD_CODE_SHOWN), fontsize=LABEL_POINTS - 7, color=INK, va="center", family="Menlo", linespacing=1.5)
    figure.text(0.05, 0.415, "What each matrix is", fontsize=LABEL_POINTS - 2, color=MUTED, va="center")
    figure.text(0.05, 0.285, "A: what a later head searches for\nB: what an earlier head writes\ncos: how well each pair lines up", fontsize=LABEL_POINTS - 2, color=INK, va="center", linespacing=1.4)
    figure.text(0.665, 0.855, "Its cosine table, top 7", fontsize=LABEL_POINTS - 2, color=MUTED, va="center")
    figure.text(0.665, 0.79, "later   earlier cos", fontsize=LABEL_POINTS - 5, color=MUTED, va="center", family="Menlo")
    for index, row in enumerate(table):
        chosen = row[0] in answer
        height = 0.725 - index * 0.068
        figure.text(0.665, height, f"{row[0]:<7} {row[1]:<7} {float(row[2]):.3f}", fontsize=LABEL_POINTS - 5, color=BLUE if chosen else INK, fontweight="bold" if chosen else "normal", va="center", family="Menlo")
    figure.text(0.665, 0.235, "Blue: its final answer.\nAll 5 are induction heads.", fontsize=LABEL_POINTS - 2, color=INK, va="center", linespacing=1.3)
    figure.text(0.5, 0.075, "Scrambled labels: 4.9 is GPT-2's head 4.11. 5.1 and 6.9 keep their real numbers by chance.", fontsize=LABEL_POINTS - 2, ha="center", va="center", color=MUTED)
    output = ROOT / "figures" / "how_astra_found_them_v1.png"
    figure.savefig(output, dpi=DPI)
    print(output, answer)


def main() -> None:
    plt.rcParams["font.family"] = "Helvetica"
    shuffle = TokenShuffle()
    _story_figure(shuffle)
    _method_figure()
    _arithmetic_figure(shuffle)
    _heads_figure(shuffle)


if __name__ == "__main__":
    main()
