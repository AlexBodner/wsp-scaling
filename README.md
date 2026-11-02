# Writing LLM weights

Can a frontier LLM, with no tools, write GPT-2 weights that do a specified task, and predict what those weights will do? This tests a stronger version of the claim in [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).

The method is settled and the pipeline passes its dry run. No Astra call has been made yet, so there are no results on Astra.

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
