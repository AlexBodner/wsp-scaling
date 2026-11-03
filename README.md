# Writing LLM weights

Can a frontier LLM, with no tools, write GPT-2 weights that do a specified task, and read the weights of a real one? This tests a stronger version of the claim in [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

Results so far, on `gpt-6-astra` with no tools, one run per arm:
- **Writing:** into an empty GPT-2, it wrote networks that reverse strings (100%) and name capitals (98.0% on the phrasing it was shown, 53.6% on hidden ones). These are small circuits, and deleting layers does not change their scores.
- **Reading:** on a disguised real GPT-2, asked to name the heads that most look back at repeated text, 0 of 10 picks were right over two reads, with a budget of 20,000 values each. Told which statistic to compute and which layers to compare, 4 of 5 picks were right, and the results it typed matched code at a correlation of 0.998. No tool calls appear in our logs, but its reasoning is encrypted, so we cannot say how it got them.

- [METHOD.md](METHOD.md): the current design.
- [DECISIONS.md](DECISIONS.md): what Alex asked for and what was decided.
- [CONTEXT.md](CONTEXT.md): the post, the objection to it, and what its logs show.
- `checks/`: the scripts behind the architecture and tokenizer numbers.
- `wsp/`, `scripts/`, `tests/`: the pipeline. `prompts/` holds the frozen prompts, and `runs/` holds every run.

## Running it
The downloads below go in `data/`, which is not tracked: GPT-2 from Hugging Face, `TinyStoriesV2-GPT4-valid.txt` as `data/tinystories/valid.txt`, the first 100 MB of `TinyStoriesV2-GPT4-train.txt` as `data/tinystories/train_first100mb.txt`, and the post's Breakout files in `data/control/`. MinAtar 1.0.15 goes in `.deps/`.

```bash
PYTHONPATH=. python3 -m pytest tests
```

```bash
PYTHONPATH=. python3 scripts/run_experiment.py capitals --reply-file prompts/hand_capitals_reply.md --label hand
```

Live calls use `--live` and need `vlm-jigsaw/src` on `PYTHONPATH`. Every call is saved in full under the run's `calls/` folder.

Three things are left out of this public copy of the logs: the control call's prompt, which holds GPL-licensed MinAtar source (its hash is in `prompts/MANIFEST.json`), the post's published Breakout answer, and Codex's model catalog (its hash is in each call). Codex's session files stay on the machine that ran them, and each call's `seal` records what the scan of its file found.
