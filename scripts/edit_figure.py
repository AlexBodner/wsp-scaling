"""Draws the post figures for Astra's weight edits: real outputs before and after, and the trade-off against simpler edits.

Every word and number is read from the saved outputs of checks/edit_demo.py and checks/write_baselines.py.
"""

import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "checks" / "outputs" / "edit_demo.txt"
BASELINES = ROOT / "checks" / "outputs" / "write_baselines.txt"
BLUE, RED, GREEN, GREY, INK, MUTED = "#2a78d6", "#c8352b", "#1f8a5b", "#8f8e88", "#0b0b0b", "#52514e"
DPI = 200
TITLE_POINTS, PROMPT_POINTS, LINE_POINTS, NOTE_POINTS = 27, 23, 22, 18
# Prompt, the words marked on the unedited and edited lines, the edited line's colour, and how many words to show.
EXAMPLES = (
    ("apple, chair, river, apple, chair,", " river,", " apple,", RED, 3),
    ("red fox, blue car, green hat, red fox, blue car,", " green hat,", " blue fox,", RED, 3),
    ("The old man walked to the market and bought three fish. The old man walked to the market and bought", " three fish.", " three fish.", GREEN, 2),
    ("Once upon a time", None, None, None, 8),
)
SHOWN_RUN = "Astra, run 2"
WRAP_AT = 60
LONG_PROMPT_POINTS = 20


def sections() -> dict[str, list[str]]:
    """Returns the lines of the saved demo, by arm."""
    found: dict[str, list[str]] = {}
    for line in DEMO.read_text().splitlines():
        if line.startswith("[") and line.endswith("]"):
            found[line[1:-1]] = []
        elif line and found:
            found[list(found)[-1]].append(line)
    return found


def continuation(lines: list[str], prompt: str) -> str:
    for line in lines:
        if line.startswith(f"  {prompt!r} -> "):
            return re.sub(r"\s+", " ", ast.literal_eval(line.split(" -> ", 1)[1]))
    raise ValueError(f"no saved continuation for {prompt!r}")


def wrap(text: str) -> str:
    """Returns a long prompt split over two lines, at its sentence break."""
    return text.replace(". The", ".\nThe") if len(text) >= WRAP_AT else text


def short(text: str, words: int) -> str:
    """Returns the first words of a continuation, marked as cut when anything was dropped."""
    parts = text.split(" ")
    kept = " ".join(parts[: words + 1])
    return kept + " ..." if len(parts) > words + 1 else kept


def before_after() -> None:
    arms = sections()
    figure = plt.figure(figsize=(10.8, 10.2), facecolor="white")
    figure.text(0.04, 0.95, "GPT-2 before and after Astra's weight edit", fontsize=TITLE_POINTS, fontweight="bold", color=INK, va="center")
    top, block = 0.875, 0.2
    for index, (prompt, right, wrong, after_colour, words) in enumerate(EXAMPLES):
        y = top - index * block
        figure.text(0.04, y, wrap(f"\u201c{prompt}\u201d"), fontsize=PROMPT_POINTS if len(prompt) < WRAP_AT else LONG_PROMPT_POINTS, color=INK, va="center", fontweight="bold", linespacing=1.15)
        before = short(continuation(arms["unedited"], prompt), words)
        after = short(continuation(arms[SHOWN_RUN], prompt), words)
        for row, (name, text, colour, swatch) in enumerate((("GPT-2", before, GREEN, GREY), ("Edited", after, after_colour, BLUE))):
            height = y - (0.062 if len(prompt) < WRAP_AT else 0.075) - row * 0.052
            figure.add_artist(Rectangle((0.04, height - 0.01), 0.02, 0.02, transform=figure.transFigure, facecolor=swatch, linewidth=0))
            figure.text(0.07, height, name, fontsize=LINE_POINTS, color=MUTED, va="center")
            marked = right if row == 0 else wrong
            if marked is None:
                figure.text(0.2, height, text.strip(), fontsize=LINE_POINTS, color=INK, va="center")
            else:
                label = figure.text(0.2, height, marked.strip(), fontsize=LINE_POINTS, color=colour, fontweight="bold", va="center")
                figure.canvas.draw()
                box = label.get_window_extent().transformed(figure.transFigure.inverted())
                figure.text(box.x1 + 0.008, height, text[len(marked):].strip(), fontsize=LINE_POINTS, color=INK, va="center")
    figure.text(0.04, 0.03, "Astra's run 2. Top next words, cut short. Green: right. Red: wrong.", fontsize=NOTE_POINTS, color=MUTED, va="center")
    figure.savefig(ROOT / "figures" / "edit_before_after_v1.png", dpi=DPI)
    plt.close(figure)


def scored(lines: list[str]) -> tuple[float, float]:
    match = re.match(r"scored test: repeat accuracy ([\d.]+), text loss ([\d.]+)", lines[0])
    return float(match[1]), float(match[2])


def trade_off() -> None:
    arms = sections()
    base = scored(arms["unedited"])[1]
    point = lambda name: (scored(arms[name])[1] - base, 100 * scored(arms[name])[0])
    layers = re.search(r"layers 5 to 7 zeroed: repeat accuracy ([\d.]+), text loss [\d.]+, rise ([\d.]+)", BASELINES.read_text())
    figure, axis = plt.subplots(figsize=(10.8, 7.4), facecolor="white")
    figure.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.14)
    figure.text(0.04, 0.93, "A simple edit matches Astra's", fontsize=TITLE_POINTS, fontweight="bold", color=INK, va="center")
    grey_points = {
        "Unedited": (point("unedited"), (0.05, 96)),
        "Top 5 heads zeroed": (point("top 5 zeroed"), (0.72, 36)),
        "Layers 5 to 7 zeroed": ((float(layers[2]), 100 * float(layers[1])), (0.25, 0.5)),
    }
    for label, ((x, y), (text_x, text_y)) in grey_points.items():
        axis.scatter(x, y, s=190, color=GREY, zorder=3, linewidth=0)
        axis.text(text_x, text_y, label, fontsize=LINE_POINTS, color=MUTED, va="center")
    for name in ("head list, run 1's heads", "head list, run 2's heads"):
        axis.scatter(*point(name), s=190, facecolor="white", edgecolor=INK, linewidth=2.5, zorder=4)
    for name in ("Astra, run 1", "Astra, run 2"):
        axis.scatter(*point(name), s=190, color=BLUE, zorder=5, linewidth=0)
    axis.annotate("Astra, run 1", point("Astra, run 1"), (0.07, 20.5), fontsize=LINE_POINTS, color=BLUE, fontweight="bold", va="center")
    axis.annotate("Astra, run 2", point("Astra, run 2"), (0.07, 13.5), fontsize=LINE_POINTS, color=BLUE, fontweight="bold", va="center", arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 1.5, "shrinkA": 3, "shrinkB": 6, "relpos": (0, 0.5)})
    axis.annotate("Simple edit, same heads", point("head list, run 2's heads"), (0.07, 6.5), fontsize=LINE_POINTS, color=INK, va="center", arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 1.5, "shrinkA": 3, "shrinkB": 6, "relpos": (0, 0.5)})
    axis.set_xlim(-0.04, 1.2)
    axis.set_ylim(-4, 106)
    axis.set_xlabel("Damage to ordinary text (rise in loss)", fontsize=LINE_POINTS, color=INK, labelpad=10)
    axis.set_ylabel("Random-token repeats still right (%)", fontsize=LINE_POINTS, color=INK, labelpad=10)
    axis.tick_params(labelsize=NOTE_POINTS + 1, colors=MUTED, length=0)
    axis.set_xticks([0, 0.5, 1])
    axis.set_yticks([0, 50, 100])
    axis.grid(color="#ecebe6", linewidth=1)
    axis.set_axisbelow(True)
    for side in axis.spines.values():
        side.set_visible(False)
    figure.savefig(ROOT / "figures" / "edit_trade_off_v1.png", dpi=DPI)
    plt.close(figure)


if __name__ == "__main__":
    before_after()
    trade_off()
