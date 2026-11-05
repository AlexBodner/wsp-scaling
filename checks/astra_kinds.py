"""Scores Astra's runs on other kinds of head, against where heads attend when GPT-2 runs. Makes no Astra calls.

Picks are mapped to real heads through each run's scramble. Chance is a permutation of each pick within its own layer,
because the scramble hides heads and keeps layers. Baselines are simple statistics, on all heads and within the best
layers, which is an upper bound for what a layer memory plus one line of code reaches.
"""
import json
import re
from pathlib import Path

import numpy as np

from checks.head_types import weight_statistics
from wsp.disguise import HEADS, LAYERS
from wsp.head_kinds import NAMED, QUESTIONS, kind_scores
from wsp.network import TokenShuffle, real_network, writable_tensors
from wsp.reading import head_scores

ROOT = Path(__file__).parent.parent
PERMUTATIONS = 20000
NEAR = 0.02
SEED = 20261104
_GROUPS = re.compile(r"```groups[ \t]*\n(.*?)```", re.DOTALL)
_GROUP_LINE = re.compile(r"^\s*(\d+)\.(\d+)\s*\|\s*(.+?)\s*\|\s*([\d.]+)\s*$")


def latest_run(kind: str) -> Path | None:
    """Returns a kind's latest finished run. A run stopped by a failure has no result file and no score."""
    runs = [run for run in sorted((ROOT / "runs").glob(f"kind-{kind}__live__*")) if (run / "result.json").exists()]
    return runs[-1] if runs else None


def permutation_chance(picks: list[tuple[int, int]], top: set[tuple[int, int]], rng: np.random.Generator) -> tuple[float, float]:
    """Returns the mean overlap and the chance of matching or beating the real overlap, with picks shuffled within layers."""
    real = sum(pick in top for pick in picks)
    counts = np.zeros(PERMUTATIONS)
    for i in range(PERMUTATIONS):
        taken: set[tuple[int, int]] = set()
        for layer, _ in picks:
            options = [(layer, head) for head in range(HEADS) if (layer, head) not in taken]
            taken.add(options[int(rng.integers(len(options)))])
        counts[i] = len(taken & top)
    return float(counts.mean()), float((counts >= real).mean())


def baselines(tensors: dict, values: np.ndarray, top: set[int]) -> str:
    """Returns the overlap of each simple statistic with a kind's top 5, on all heads and within the kind's own layers."""
    cells = []
    layers = {index // HEADS for index in top}
    allowed = np.array([index // HEADS in layers for index in range(LAYERS * HEADS)])
    for label, statistic in weight_statistics(tensors).items():
        overall = len(set(np.argsort(-statistic)[:NAMED].tolist()) & top)
        within = len(set(np.argsort(-np.where(allowed, statistic, -np.inf))[:NAMED].tolist()) & top)
        cells.append(f"{label} {overall} ({within} in its layers)")
    return "; ".join(cells)


def score_open(run: Path, scores: dict[str, np.ndarray], original: np.ndarray) -> None:
    """Prints where the heads in each of the open run's groups attend, and how each kind's top heads were grouped."""
    replies = [json.loads(line)["reply"] for line in (run / "transcript.jsonl").read_text().splitlines()]
    found = _GROUPS.findall(replies[-1])
    if not found:
        print("  no groups block in the final reply")
        return
    groups: dict[str, list[tuple[tuple[int, int], float]]] = {}
    for line in found[-1].splitlines():
        match = _GROUP_LINE.match(line)
        if match:
            layer, head = int(match[1]), int(match[2])
            groups.setdefault(match[3].lower(), []).append(((layer, int(original[layer][head])), float(match[4])))
    print(f"  {sum(map(len, groups.values()))} heads placed in {len(groups)} groups")
    for job, members in sorted(groups.items(), key=lambda item: -len(item[1])):
        means = {kind: float(np.mean([values[head] for head, _ in members])) for kind, values in scores.items()}
        best = max(means, key=means.get)
        print(f"  {job!r} ({len(members)} heads, confidence {np.mean([c for _, c in members]):.2f}): highest on {best} {means[best]:.2f}; " + " ".join(f"{kind} {value:.2f}" for kind, value in means.items()))
    rng = np.random.default_rng(SEED)
    placements = [(head, job) for job, members in groups.items() for head, _ in members]
    for kind, values in scores.items():
        top = np.argsort(-values.ravel())[:NAMED]
        heads = [(int(i) // HEADS, int(i) % HEADS) for i in top]
        counts = {job: sum(head in {member for member, _ in members} for head in heads) for job, members in groups.items()}
        job = max(counts, key=counts.get)
        placed = sum(counts.values())
        most = np.zeros(PERMUTATIONS // 10)
        for trial in range(len(most)):
            shuffled = {}
            for layer in range(LAYERS):
                in_layer = [job_name for (head, job_name) in placements if head[0] == layer]
                chosen = rng.permutation(HEADS)[: len(in_layer)]
                for head, job_name in zip(chosen, in_layer):
                    shuffled[(layer, int(head))] = job_name
            jobs = [shuffled.get(head) for head in heads]
            most[trial] = max((jobs.count(name) for name in set(jobs) - {None}), default=0)
        print(f"  top 5 by {kind}: {placed} placed, {counts[job]} in one group, {job!r}; within-layer shuffle gets that many or more {float((most >= counts[job]).mean()):.3f}")
    for job, members in sorted(groups.items(), key=lambda item: -len(item[1])):
        best = sorted(members, key=lambda member: -member[1])[:6]
        print(f"  {job!r} top members, real heads: " + ", ".join(f"{head[0]}.{head[1]} ({c:.2f}; prev {scores['previous'][head]:.2f}, dup {scores['duplicate'][head]:.2f}, self {scores['self'][head]:.2f}, ind {scores['induction'][head]:.2f}, nm {scores['name_mover'][head]:.2f}, si {scores['s_inhibition'][head]:.2f})" for head, c in best))


def main() -> None:
    shuffle = TokenShuffle()
    model = real_network(shuffle).double()
    model.set_attn_implementation("eager")
    scores = {kind: values for kind, values in kind_scores(model, shuffle).items()}
    induction = head_scores(model)
    tensors = {key: value.detach().numpy().astype(np.float64) for key, value in writable_tensors(model).items()}
    rng = np.random.default_rng(SEED)
    for kind in QUESTIONS:
        values = scores[kind]
        order = np.argsort(-values.ravel())
        top = {(int(i) // HEADS, int(i) % HEADS) for i in order[:NAMED]}
        fifth = values.ravel()[order[NAMED - 1]]
        print(f"{kind}: truth " + ", ".join(f"{int(i) // HEADS}.{int(i) % HEADS} {values.ravel()[i]:.2f}" for i in order[:NAMED + 2]))
        print(f"  baselines: {baselines(tensors, values, {int(i) for i in order[:NAMED]})}")
        run = latest_run(kind)
        if run is None:
            print("  no live run yet")
            continue
        result = json.loads((run / "result.json").read_text())
        original = json.loads((run / "disguise_map.json").read_text())
        picks = [(layer, int(original[layer][head])) for layer, head in result["answer"] or []]
        hits = sum(pick in top for pick in picks)
        near = sum(values[pick] >= fifth - NEAR for pick in picks)
        chance, tail = permutation_chance(picks, top, rng) if picks else (0.0, 1.0)
        ranks = [int(np.where(order == layer * HEADS + head)[0][0]) + 1 for layer, head in picks]
        print(f"  Astra: {hits} of {NAMED}, {near} within {NEAR} of the 5th, picks {picks}, ranks {ranks}, turns {result['turns']}, code runs {result['code_runs']}, flags {result['forward_pass_flags']}, stopped by {result['stopped_by']}")
        print(f"  chance within layers: mean {chance:.2f}, p {tail:.4f}")
    run = latest_run("open")
    print("open: sort the heads")
    if run is None:
        print("  no live run yet")
    else:
        all_scores = {**scores, "induction": induction}
        score_open(run, all_scores, np.array(json.loads((run / "disguise_map.json").read_text())))


if __name__ == "__main__":
    main()
