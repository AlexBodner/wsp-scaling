"""Draws the post figures for Astra's weight edits: what it was given, the table it chose heads from, real outputs before and after, and the same simple edit on different heads.

Every result is read from the saved outputs of checks/edit_demo.py, checks/edit_controls.py and checks/write_baselines.py or from the runs' own logs, and the words marked in colour are checked against them.
"""

import ast
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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
RUNS = {"Run 1": ROOT / "runs/write-write-1__live__20261104T053859Z", "Run 2": ROOT / "runs/write-write-2__live__20261104T061358Z"}
TOP_FIVE = {(5, 5), (7, 10), (6, 9), (5, 1), (7, 2)}
SOURCE = (4, 11)
TABLE_ROWS = 10


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




def box(figure, x: float, y: float, width: float, height: float, text: str, face: str, edge: str, colour: str = INK, weight: str = "normal") -> None:
    figure.add_artist(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=0.012", transform=figure.transFigure, facecolor=face, edgecolor=edge, linewidth=2))
    figure.text(x + width / 2, y + height / 2, text, fontsize=LINE_POINTS - 1, color=colour, ha="center", va="center", fontweight=weight, linespacing=1.3)


def arrow(figure, start: tuple[float, float], end: tuple[float, float], style: str = "solid") -> None:
    figure.add_artist(FancyArrowPatch(start, end, transform=figure.transFigure, arrowstyle="-|>", mutation_scale=26, color=MUTED, linewidth=2.5, linestyle=style))


def what_it_got() -> None:
    """Draws what went into the task and what came out, with the part Astra never saw set apart."""
    figure = plt.figure(figsize=(10.8, 10.2), facecolor="white")
    figure.text(0.04, 0.95, "What Astra was given", fontsize=TITLE_POINTS, fontweight="bold", color=INK, va="center")
    pale, line = "#eef4fb", BLUE
    box(figure, 0.04, 0.72, 0.29, 0.17, "The network's\nformula, written\nout in full", pale, line)
    box(figure, 0.355, 0.72, 0.29, 0.17, "85M weights,\nheads scrambled,\nno word tables", pale, line)
    box(figure, 0.67, 0.72, 0.29, 0.17, "The goal: stop\nfinishing repeats,\nkeep English", pale, line)
    for x in (0.185, 0.5, 0.815):
        arrow(figure, (x, 0.715), (0.5 if x == 0.5 else x * 0.6 + 0.2, 0.635))
    box(figure, 0.2, 0.5, 0.6, 0.13, "GPT-6 Astra with Python\nmath on the weights, no running it", BLUE, BLUE, "white", "bold")
    arrow(figure, (0.5, 0.495), (0.5, 0.42))
    box(figure, 0.2, 0.29, 0.6, 0.125, "It returned: code that edits\nthe weights, and 5 head names", pale, line)
    arrow(figure, (0.5, 0.285), (0.5, 0.21), "dashed")
    box(figure, 0.2, 0.08, 0.6, 0.125, "We ran the edited model and scored it.\nAstra never saw this step.", "#f2f1ec", GREY, MUTED)
    figure.text(0.04, 0.035, "Also in the prompt: the exact weight count, \u201cName 5 heads\u201d, and that matrix\nproducts are allowed. It never names the model.", fontsize=NOTE_POINTS - 1, color=MUTED, va="center", linespacing=1.3)
    figure.savefig(ROOT / "figures" / "edit_what_it_got_v1.png", dpi=DPI)
    plt.close(figure)


def printed_tables() -> dict[str, list[tuple[tuple[int, int], float]]]:
    """Returns each run's first table ranking heads against the source, from its logged output, in real head numbers."""
    tables = {}
    for name, run in RUNS.items():
        turns = [json.loads(line) for line in (run / "transcript.jsonl").read_text().splitlines()]
        order = json.loads((run / "disguise_map.json").read_text())
        if name == "Run 1":
            output = turns[0]["output"]
            pairs = re.findall(r"target (\d+)\.(\d+) <- source (\d+)\.(\d+): ([\d.]+);", output[output.index("Largest positive") :])
            rows = [((int(a), int(b)), float(value)) for a, b, c, d, value in pairs if (int(c), order[int(c)][int(d)]) == SOURCE]
        else:
            output = turns[2]["output"]
            found = re.findall(r"^(\d+)\.(\d+)\s+[-\d.]+\s+[-\d.]+\s+([-\d.]+)\s+[-\d.]+\s+[-\d.]+\s+\S+$", output[output.index("TARGETS:") : output.index("SPECTRA")], re.M)
            rows = sorted((((int(a), int(b)), float(value)) for a, b, value in found), key=lambda row: -row[1])
        tables[name] = [((layer, order[layer][head]), value) for (layer, head), value in rows[:TABLE_ROWS]]
    return tables


def edited_heads(run: Path) -> set[tuple[int, int]]:
    order = json.loads((run / "disguise_map.json").read_text())
    return {(layer, order[layer][head]) for layer, head in json.loads((run / "write_score.json").read_text())["touched_heads"]}


def how_it_found() -> None:
    """Draws the table each run chose heads from, with the true top five marked, which Astra could not see."""
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 8.6), facecolor="white")
    figure.subplots_adjust(left=0.09, right=0.97, top=0.76, bottom=0.14, wspace=0.42)
    figure.text(0.04, 0.945, "Its printed numbers put the true top 5 highest", fontsize=TITLE_POINTS, fontweight="bold", color=INK, va="center")
    figure.text(0.04, 0.875, "How closely each head's search pattern matches\nwhat one earlier head writes, from Astra's code. We sorted run 2's.", fontsize=LINE_POINTS - 2, color=MUTED, va="center", linespacing=1.3)
    tables = printed_tables()
    for axis, (name, rows) in zip(axes, tables.items()):
        if {head for head, _ in rows[:len(TOP_FIVE)]} != TOP_FIVE:
            raise ValueError(f"{name}'s table no longer opens with the top five")
        edited = edited_heads(RUNS[name])
        positions = list(range(len(rows)))[::-1]
        axis.barh(positions, [value for _, value in rows], height=0.72, color=[BLUE if head in TOP_FIVE else "#c9c8c2" for head, _ in rows], linewidth=0)
        axis.set_yticks(positions)
        axis.set_yticklabels([f"{layer}.{head}" for (layer, head), _ in rows], fontsize=LINE_POINTS - 2, color=INK)
        for position, (head, value) in zip(positions, rows):
            if head in edited:
                axis.text(value + rows[0][1] * 0.03, position, "edited", fontsize=NOTE_POINTS, color=INK, va="center")
        axis.set_xlim(0, rows[0][1] * 1.42)
        axis.set_title(name, fontsize=LINE_POINTS, fontweight="bold", color=INK, loc="left", pad=10)
        axis.set_xticks([])
        axis.tick_params(length=0)
        for side in axis.spines.values():
            side.set_visible(False)
    figure.add_artist(Rectangle((0.04, 0.075), 0.022, 0.022, transform=figure.transFigure, facecolor=BLUE, linewidth=0))
    figure.text(0.07, 0.086, "True top 5 by attention on repeats, measured by running the model", fontsize=NOTE_POINTS, color=INK, va="center")
    figure.text(0.04, 0.035, "Real head names added by us. Astra saw scrambled ones. Scales differ by run.", fontsize=NOTE_POINTS - 1, color=MUTED, va="center")
    figure.savefig(ROOT / "figures" / "edit_how_it_found_v2.png", dpi=DPI)
    plt.close(figure)


def logged_line(turns: list[dict], turn: int, field: str, start: str, end: str) -> str:
    """Returns the stretch of a logged turn from start to end, and fails if the log no longer holds it."""
    text = re.sub(r"\s+", " ", turns[turn - 1][field]).replace("\u2019", "'")
    first = text.index(start)
    return text[first : text.index(end, first) + len(end)]


def its_words() -> None:
    """Draws Astra's own words and code from run 1 beside our reading of them, which are labelled as ours."""
    turns = [json.loads(line) for line in (RUNS["Run 1"] / "transcript.jsonl").read_text().splitlines()]
    rows = (
        ("Its first message", "\u201c" + logged_line(turns, 1, "reply", "whether its query/key", "value/output matrix") + "\u201d", "serif", "It looks for the two-step circuit\nbefore seeing any numbers."),
        ("Its first code", "A = Q @ K.T\nB = V @ O\nalign = (AA @ BB.T) / ...", "monospace", "What a later head searches for, what\nan earlier head writes, how alike."),
        ("What that printed", logged_line(turns, 1, "output", "target 5.6 <- source 4.6", "0.53622") + "\n" + logged_line(turns, 1, "output", "target 6.4 <- source 4.6", "0.49798"), "monospace", "One earlier head tops the list.\nIn real numbering: 5.1, 6.9 and 4.11."),
        ("Its edit plan", "\u201c" + logged_line(turns, 5, "output", "Edits cancel source-4.6", "query-bias term.") + "\u201d", "serif", "Break the match but keep each\nhead's default, so other text survives."),
        ("Its last message", "\u201c" + logged_line(turns, 10, "reply", "Their effects", "remain unmeasured.") + "\u201d", "serif", "It never saw whether this worked."),
    )
    figure = plt.figure(figsize=(10.8, 10.8), facecolor="white")
    figure.text(0.04, 0.955, "What Astra wrote, and our reading of it", fontsize=TITLE_POINTS, fontweight="bold", color=INK, va="center")
    figure.text(0.04, 0.905, "Run 1. Its reasoning is encrypted, so the reading on the right is ours.", fontsize=NOTE_POINTS, color=MUTED, va="center")
    figure.text(0.04, 0.855, "Astra's words", fontsize=NOTE_POINTS, color=BLUE, fontweight="bold", va="center")
    figure.text(0.57, 0.855, "Our reading", fontsize=NOTE_POINTS, color=MUTED, fontweight="bold", va="center")
    for index, (label, words, family, reading) in enumerate(rows):
        y = 0.79 - index * 0.16
        figure.add_artist(plt.Line2D([0.04, 0.96], [y + 0.04, y + 0.04], color="#dddcd6", linewidth=1, transform=figure.transFigure))
        figure.text(0.04, y + 0.012, label, fontsize=NOTE_POINTS - 2, color=MUTED, va="center")
        wrapped = "\n".join(textwrap.wrap(words, 38)) if family == "serif" else words
        figure.text(0.04, y - 0.05, wrapped, fontsize=NOTE_POINTS - (0 if family == "serif" else 2), color=INK, family=family, va="center", linespacing=1.25)
        figure.text(0.57, y - 0.03, reading, fontsize=NOTE_POINTS - 1, color=INK, va="center", linespacing=1.3)
    figure.savefig(ROOT / "figures" / "edit_its_words_v1.png", dpi=DPI)
    plt.close(figure)


def what_works() -> None:
    """Draws every edit on GPT-2 as a table of what it did to repeats and to ordinary text."""
    base = scored(sections()["unedited"])[1]
    arms = sections()
    controls = CONTROLS.read_text()
    control = lambda start: control_point(start, base)
    astra = lambda name: (100 * scored(arms[name])[0], 100 * (scored(arms[name])[1] - base) / base)
    swap = lambda point: (point[1], point[0])
    layers = re.search(r"layers 5 to 7 zeroed: repeat accuracy ([\d.]+), text loss [\d.]+, rise ([\d.]+)", BASELINES.read_text())
    top_zeroed = astra("top 5 zeroed")
    randoms = [control_point(f"random 5 {pool}, set {number}", base) for pool in ("anywhere, not the top ten", "in layers 5 to 7, not the top ten") for number in range(RANDOM_SETS)]
    mean_fixed = re.search(r"top 5 mean-ablated, uses a forward pass: repeat ([\d.]+), text rise \+?([\d.-]+)", CONTROLS.read_text())
    rows = (
        ("Astra's edits", None, None, True),
        ("Run 1", astra("Astra, run 1"), None, False),
        ("Run 2", astra("Astra, run 2"), None, False),
        ("The same simple edit", None, None, True),
        ("On the top 5 heads", swap(control("top 5:")), None, False),
        ("On 16 random sets of 5", (min(p[1] for p in randoms), max(p[1] for p in randoms)), (min(p[0] for p in randoms), max(p[0] for p in randoms)), False),
        ("On all of layers 5 to 7", swap(control("all 36 heads of layers 5 to 7")), None, False),
        ("Blunter edits", None, None, True),
        ("Zero the top 5's outputs", top_zeroed, None, False),
        ("Zero all of layers 5 to 7", (100 * float(layers[1]), 100 * float(layers[2]) / base), None, False),
        ("Needs running the model, so Astra couldn't", None, None, True),
        ("Fix the top 5 at their mean output", (100 * float(mean_fixed[1]), 100 * float(mean_fixed[2]) / base), None, False),
    )
    figure = plt.figure(figsize=(10.8, 12.2), facecolor="white")
    figure.text(0.04, 0.958, "The list of heads does much of the work", fontsize=TITLE_POINTS, fontweight="bold", color=INK, va="center")
    figure.text(0.04, 0.91, "Edits on GPT-2. Good is few repeats and little text damage.", fontsize=NOTE_POINTS + 1, color=MUTED, va="center")
    figure.text(0.67, 0.848, "Repeats\nstill right", fontsize=NOTE_POINTS, color=MUTED, ha="center", va="center", linespacing=1.15)
    figure.text(0.87, 0.848, "Text loss\nrise", fontsize=NOTE_POINTS, color=MUTED, ha="center", va="center", linespacing=1.15)
    y = 0.79
    for label, first, second, heading in rows:
        if heading:
            figure.add_artist(plt.Line2D([0.04, 0.96], [y + 0.024, y + 0.024], color="#dddcd6", linewidth=1, transform=figure.transFigure))
            figure.text(0.04, y - 0.004, label, fontsize=NOTE_POINTS, color=MUTED, va="center")
            y -= 0.05
            continue
        figure.text(0.06, y, label, fontsize=LINE_POINTS - 2, color=BLUE if label.startswith("Run") else INK, fontweight="bold" if label.startswith("Run") else "normal", va="center")
        if second is None:
            repeats, damage = f"{first[0]:.0f}%", f"{first[1]:.1f}%"
        else:
            repeats, damage = f"{first[0]:.0f} to {first[1]:.0f}%", f"{second[0]:.1f} to {second[1]:.0f}%"
        figure.text(0.67, y, repeats, fontsize=LINE_POINTS - 2, color=INK, ha="center", va="center")
        figure.text(0.87, y, damage, fontsize=LINE_POINTS - 2, color=INK, ha="center", va="center")
        y -= 0.054
    figure.text(0.04, 0.07, "The simple edit needs only a list of heads, and reuses a trick from Astra's run 1.", fontsize=NOTE_POINTS - 4, color=MUTED, va="center")
    figure.text(0.04, 0.035, "Random sets exclude the top 10 heads. Text loss is on children's stories.", fontsize=NOTE_POINTS - 4, color=MUTED, va="center")
    figure.savefig(ROOT / "figures" / "edit_what_works_v1.png", dpi=DPI)
    plt.close(figure)


def side_effects() -> None:
    """Draws what the edit changed against what removing the same heads changes, as a table of saved scores."""
    effects = (ROOT / "checks" / "outputs" / "edit_side_effects.txt").read_text()
    controls = CONTROLS.read_text()
    arm = lambda label, ability: float(re.search(rf"^{re.escape(label)}: .*?{ability} ([\d.]+)", effects, re.M)[1])
    demo = sections()
    repeat = lambda name: scored(demo[name])[0]
    removed = re.search(r"top 5 mean-ablated, uses a forward pass: repeat ([\d.]+), text rise [+-][\d.]+, short lists ([\d.]+)", controls)
    short = lambda name: float(re.search(r"copying, short comma list: ([\d.]+)", "\n".join(demo[name]))[1])
    rows = [
        ("Repeating random tokens", repeat("unedited"), repeat("Astra, run 1"), repeat("Astra, run 2"), float(removed[1])),
        ("Repeating noun lists", short("unedited"), short("Astra, run 1"), short("Astra, run 2"), float(removed[2])),
    ]
    for label, ability in (("Capitals", "capitals"), ("Pronoun gender", "pronoun gender"), ("Days, months, numbers", "successor"), ("Picking a later year", "greater-than years"), ("Who received it", "indirect object")):
        rows.append((label, arm("unedited", ability), arm("Astra, run 1", ability), arm("Astra, run 2", ability), arm("top 5 fixed at their mean, real weights", ability)))
    figure = plt.figure(figsize=(10.8, 9.6), facecolor="white")
    figure.text(0.04, 0.95, "What the edit changed, and what held up", fontsize=TITLE_POINTS - 2, fontweight="bold", color=INK, va="center")
    figure.text(0.04, 0.89, "Repeats: the top word is right. The rest: the right answer beats the wrong\nones. The last column takes running the model.", fontsize=NOTE_POINTS, color=MUTED, va="center", linespacing=1.3)
    columns = (0.5, 0.63, 0.76, 0.89)
    for x, title in zip(columns, ("Before", "Astra\nrun 1", "Astra\nrun 2", "Top 5 fixed\nat average")):
        figure.text(x, 0.795, title, fontsize=NOTE_POINTS, color=MUTED, ha="center", va="center", linespacing=1.15)
    for index, (label, *values) in enumerate(rows):
        y = 0.71 - index * 0.078
        figure.add_artist(plt.Line2D([0.04, 0.96], [y + 0.039, y + 0.039], color="#dddcd6", linewidth=1, transform=figure.transFigure))
        figure.text(0.04, y, label, fontsize=LINE_POINTS - 2, color=INK, va="center", fontweight="bold" if index < 2 else "normal")
        for x, value, is_astra in zip(columns, values, (False, True, True, False)):
            fell = value < values[0] - 0.02
            figure.text(x, y, f"{100 * value:.0f}%", fontsize=LINE_POINTS - 2, color=RED if fell and is_astra else INK, fontweight="bold" if fell and is_astra else "normal", ha="center", va="center")
    figure.text(0.04, 0.095, "Fixing the top 5 at their average output drops the last two about as much.", fontsize=NOTE_POINTS - 2, color=INK, va="center")
    figure.text(0.04, 0.045, "Other repeats: names are mixed, and repeating a story helps 29 and 8 points, vs 45.", fontsize=NOTE_POINTS - 3, color=INK, va="center")
    figure.savefig(ROOT / "figures" / "edit_side_effects_v3.png", dpi=DPI)
    plt.close(figure)


def match_picture() -> None:
    """Draws what the matching test sees: what one earlier head writes, heads that do and don't search for it, and a hard case, from the weights alone."""
    data = json.loads((ROOT / "checks" / "outputs" / "edit_method_data.json").read_text())
    match, other, hard = data["heads"]["matches"], data["heads"]["doesn't match"], data["heads"]["hard case"]
    figure = plt.figure(figsize=(10.8, 10.8), facecolor="white")
    figure.text(0.04, 0.955, "How Astra's code found the heads", fontsize=TITLE_POINTS, fontweight="bold", color=INK, va="center")
    figure.text(0.04, 0.89, "Multiply a head's weights to get what it searches for and what it writes.\nNo words go in. Then compare a head's search with earlier heads' writing.", fontsize=NOTE_POINTS, color=MUTED, va="center", linespacing=1.3)
    panels = (
        ([0.04, 0.5, 0.21, 0.21], data["writes"], f"{data['source']} writes this", "a line by design", MUTED),
        ([0.275, 0.5, 0.21, 0.21], match["seen"], f"{match['head']} searches", f"match {match['cosine']:.2f}", BLUE),
        ([0.51, 0.5, 0.21, 0.21], other["seen"], f"{other['head']} searches", f"match {other['cosine']:.2f}", MUTED),
        ([0.745, 0.5, 0.21, 0.21], hard["seen"], f"{hard['head']} searches", f"match {hard['cosine']:.2f}", INK),
    )
    search_limit = float(np.percentile(np.abs(np.array(match["seen"])), 99.5))
    for box, values, title, score, colour in panels:
        axis = figure.add_axes(box)
        values = np.array(values)
        limit = float(np.percentile(np.abs(values), 99.5)) if title.endswith("writes this") else search_limit
        axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        for side in axis.spines.values():
            side.set_color(GREY)
        figure.text(box[0], 0.455, title, fontsize=NOTE_POINTS, color=INK, va="center")
        figure.text(box[0], 0.415, score, fontsize=NOTE_POINTS, color=colour, fontweight="bold", va="center")
    figure.text(0.04, 0.335, "6.9 has a line where 4.11's is. 6.3 has almost nothing there.", fontsize=LINE_POINTS - 1, color=INK, va="center")
    figure.text(0.04, 0.275, "5.8 scores 0.32 but barely does induction, so the score isn't perfect.", fontsize=LINE_POINTS - 3, color=INK, va="center")
    figure.text(0.04, 0.195, "This is the published two-step circuit for finishing A B ... A with B.\nAstra's run 1 proposed this test before seeing any output.", fontsize=NOTE_POINTS, color=MUTED, va="center", linespacing=1.3)
    figure.text(0.04, 0.08, "Real weights in 4.11's own top 32 of 64 directions, so this looks cleaner than a match of 0.50.\nThe searches share one colour scale. 6.3's values are about 40x smaller than 6.9's.\nMatch is the cosine Astra's code computes. Head names added by us.", fontsize=NOTE_POINTS - 4, color=MUTED, va="center", linespacing=1.3)
    figure.savefig(ROOT / "figures" / "edit_match_v3.png", dpi=DPI)
    plt.close(figure)


def neo_result() -> None:
    """Draws the five heads both runs on GPT-Neo edited, against every head's measured attention to the previous token, and the before and after."""
    check = (ROOT / "checks" / "outputs" / "neo_edit_check.txt").read_text()
    rows = [(name, float(value)) for name, value in re.findall(r"(\d+\.\d+) ([\d.]+)", check.split("top previous-token heads: ")[1].splitlines()[0])][:8]
    named = [set(ast.literal_eval(found)) for found in re.findall(r"named heads, real numbers: (\[.*?\])", check)]
    if len(named) != 2 or named[0] != named[1]:
        raise ValueError("the two runs no longer name the same heads")
    outcome = lambda arm: re.search(rf"\[Astra, run {arm}\]\nscored test: repeat accuracy ([\d.]+)", check)[1]
    prompt = "apple, chair, river, apple, chair,"
    said = lambda label: next(ast.literal_eval(line.split(" -> ", 1)[1]) for line in check.split(f"[{label}]")[1].splitlines() if line.startswith(f"  {prompt!r}"))
    figure = plt.figure(figsize=(10.8, 10.2), facecolor="white")
    figure.text(0.04, 0.95, "Another model: 3 of 4 runs edited the same 5 heads", fontsize=TITLE_POINTS - 3, fontweight="bold", color=INK, va="center")
    figure.text(0.04, 0.885, "GPT-Neo 125M, trained separately from GPT-2. Each head's attention\nto the previous token, measured by running the model.", fontsize=NOTE_POINTS + 1, color=MUTED, va="center", linespacing=1.3)
    axis = figure.add_axes([0.12, 0.42, 0.82, 0.41])
    positions = list(range(len(rows)))[::-1]
    axis.barh(positions, [value for _, value in rows], height=0.72, color=[BLUE if name in named[0] else "#c9c8c2" for name, _ in rows], linewidth=0)
    axis.set_yticks(positions)
    axis.set_yticklabels([name for name, _ in rows], fontsize=LINE_POINTS - 2, color=INK)
    for position, (name, value) in zip(positions, rows):
        axis.text(value + 0.012, position, "edited" if name in named[0] else "", fontsize=NOTE_POINTS, color=INK, va="center")
    axis.set_xlim(0, 1.15)
    axis.set_xticks([])
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    figure.text(0.04, 0.345, f"\u201c{prompt}\u201d", fontsize=PROMPT_POINTS, color=INK, fontweight="bold", va="center")
    for y, name, text, colour, swatch in ((0.285, "Neo", said("unedited"), GREEN, GREY), (0.23, "Edited", said("Astra, run 2"), RED, BLUE)):
        figure.add_artist(Rectangle((0.04, y - 0.01), 0.02, 0.02, transform=figure.transFigure, facecolor=swatch, linewidth=0))
        figure.text(0.07, y, name, fontsize=LINE_POINTS, color=MUTED, va="center")
        figure.text(0.2, y, short(re.sub(r"\s+", " ", text), 3), fontsize=LINE_POINTS, color=colour, fontweight="bold", va="center")
    clean = lambda arm: 100 * json.loads((ROOT / "runs" / f"write-neo-{arm}__live__20261105T002705Z" / "write_score.json").read_text())["after"][0]
    figure.text(0.04, 0.16, f"Repeats still right: 96% to {100 * float(outcome(1)):.1f}% and {100 * float(outcome(2)):.1f}%", fontsize=LINE_POINTS, color=INK, va="center")
    figure.text(0.04, 0.11, f"Rerun with no \u201cneo\u201d in its folder: {clean(1):.1f}% and {clean(2):.0f}%", fontsize=LINE_POINTS, color=INK, va="center")
    figure.text(0.04, 0.06, "Astra never ran it. The 42% run edited 5 of Neo's top 10 induction heads.", fontsize=NOTE_POINTS, color=MUTED, va="center")
    figure.text(0.04, 0.025, "Blue: the five heads the other 3 runs edited. Real head names added by us.", fontsize=NOTE_POINTS, color=MUTED, va="center")
    figure.savefig(ROOT / "figures" / "edit_neo_v1.png", dpi=DPI)
    plt.close(figure)


if __name__ == "__main__":
    what_works()
    side_effects()
    its_words()
    match_picture()
    neo_result()
    what_it_got()
    how_it_found()
    before_after()
