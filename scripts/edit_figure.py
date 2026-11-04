"""Draws the post figures for Astra's weight edits: real outputs before and after, and the same simple edit on different heads.

Every word and number is read from the saved outputs of checks/edit_demo.py, checks/edit_controls.py and checks/write_baselines.py, and the words marked in colour are checked against them.
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
CONTROLS = ROOT / "checks" / "outputs" / "edit_controls.txt"
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
RANDOM_SETS = 8


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
    figure.text(0.04, 0.95, "GPT-2 before and after the edit", fontsize=TITLE_POINTS, fontweight="bold", color=INK, va="center")
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
                if not text.startswith(marked):
                    raise ValueError(f"saved text {text!r} doesn't start with {marked!r}")
                label = figure.text(0.2, height, marked.strip(), fontsize=LINE_POINTS, color=colour, fontweight="bold", va="center")
                figure.canvas.draw()
                box = label.get_window_extent().transformed(figure.transFigure.inverted())
                figure.text(box.x1 + 0.008, height, text[len(marked):].strip(), fontsize=LINE_POINTS, color=INK, va="center")
    figure.text(0.04, 0.03, "Astra's run 2, 4 of 8 fixed prompts, cut short. Green: right. Red: wrong.\nOver 48 repeated stories, the gain from repeating falls from 45 to 8 points.", fontsize=NOTE_POINTS, color=MUTED, va="center")
    figure.savefig(ROOT / "figures" / "edit_before_after_v2.png", dpi=DPI)
    plt.close(figure)


def scored(lines: list[str]) -> tuple[float, float]:
    match = re.match(r"scored test: repeat accuracy ([\d.]+), text loss ([\d.]+)", lines[0])
    return float(match[1]), float(match[2])


def control_point(line_start: str, base: float) -> tuple[float, float]:
    """Returns one saved control as percent rise in text loss and percent of repeats right."""
    for line in CONTROLS.read_text().splitlines():
        if line.startswith(line_start):
            match = re.search(r"repeat ([\d.]+), text rise ([+-][\d.]+)", line)
            return 100 * float(match[2]) / base, 100 * float(match[1])
    raise ValueError(f"no saved control starting {line_start!r}")


def trade_off() -> None:
    arms = sections()
    base = scored(arms["unedited"])[1]
    point = lambda name: (100 * (scored(arms[name])[1] - base) / base, 100 * scored(arms[name])[0])
    layers = re.search(r"layers 5 to 7 zeroed: repeat accuracy ([\d.]+), text loss [\d.]+, rise ([\d.]+)", BASELINES.read_text())
    figure, axis = plt.subplots(figsize=(10.8, 7.8), facecolor="white")
    figure.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.14)
    figure.text(0.04, 0.93, "The same simple edit on different heads", fontsize=TITLE_POINTS, fontweight="bold", color=INK, va="center")
    randoms = [control_point(f"random 5 {pool}, set {number}", base) for pool in ("anywhere, not the top ten", "in layers 5 to 7, not the top ten") for number in range(RANDOM_SETS)]
    axis.scatter(*zip(*randoms), s=120, facecolor="white", edgecolor=INK, linewidth=2, zorder=3)
    axis.text(9.5, 91, "On 16 random sets of 5, no top-10 head", fontsize=LINE_POINTS, color=INK, va="center")
    whole = control_point("all 36 heads of layers 5 to 7", base)
    axis.scatter(*whole, s=190, facecolor="white", edgecolor=INK, linewidth=2.5, zorder=4)
    axis.text(17, 8, "On all of layers 5 to 7", fontsize=LINE_POINTS, color=INK, va="center")
    top = control_point("top 5:", base)
    axis.scatter(*top, s=190, facecolor="white", edgecolor=INK, linewidth=2.5, zorder=5)
    axis.annotate("On the top 5", top, (5.5, 15), fontsize=LINE_POINTS, color=INK, va="center", arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 1.5, "shrinkA": 3, "shrinkB": 6, "relpos": (0, 0.5)})
    zeroed_layers = (100 * float(layers[2]) / base, 100 * float(layers[1]))
    for label, (x, y), (text_x, text_y) in (("Unedited", point("unedited"), (0.8, 106)), ("Top 5 outputs zeroed", point("top 5 zeroed"), (23, 37)), ("Layers 5 to 7\noutputs zeroed", zeroed_layers, (7.6, 2.5))):
        axis.scatter(x, y, s=190, color=GREY, zorder=3, linewidth=0)
        axis.text(text_x, text_y, label, fontsize=LINE_POINTS, color=MUTED, va="center")
    for name in ("Astra, run 1", "Astra, run 2"):
        axis.scatter(*point(name), s=190, color=BLUE, zorder=6, linewidth=0)
    axis.annotate("Astra, run 1", point("Astra, run 1"), (5.5, 34), fontsize=LINE_POINTS, color=BLUE, fontweight="bold", va="center", arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 1.5, "shrinkA": 3, "shrinkB": 6, "relpos": (0, 0.5)})
    axis.annotate("Astra, run 2", point("Astra, run 2"), (5.5, 24.5), fontsize=LINE_POINTS, color=BLUE, fontweight="bold", va="center", arrowprops={"arrowstyle": "-", "color": BLUE, "linewidth": 1.5, "shrinkA": 3, "shrinkB": 6, "relpos": (0, 0.5)})
    axis.set_xlim(-1.5, 36)
    axis.set_ylim(-5, 113)
    axis.set_xlabel("Rise in loss on ordinary text (%)", fontsize=LINE_POINTS, color=INK, labelpad=10)
    axis.set_ylabel("Random-token repeats still right (%)", fontsize=LINE_POINTS, color=INK, labelpad=10)
    axis.tick_params(labelsize=NOTE_POINTS + 1, colors=MUTED, length=0)
    axis.set_xticks([0, 10, 20, 30])
    axis.set_yticks([0, 50, 100])
    axis.grid(color="#ecebe6", linewidth=1)
    axis.set_axisbelow(True)
    for side in axis.spines.values():
        side.set_visible(False)
    figure.savefig(ROOT / "figures" / "edit_trade_off_v2.png", dpi=DPI)
    plt.close(figure)


if __name__ == "__main__":
    before_after()
    trade_off()
