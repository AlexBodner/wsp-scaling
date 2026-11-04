# What later experiments found

These follow the [reading result](README.md) and ask how it works, what else can be read, whether reading can become writing, and what changes at a bigger size. Most results had a prediction written before they ran, and a separate agent reran or recomputed them, which often narrowed the claim. Those predictions and scripts aren't public yet. Several predictions failed, and those are marked. Astra's runs are in `runs/`, each one call log per turn. Totals are from one model and few runs, so treat them as leads rather than settled facts.

## Reading can become writing, on GPT-2 small
Asked to make the model stop repeating while keeping ordinary text, with Python, no forward pass and scrambled weights, Astra wrote edits to the weights and never saw a score. Both runs cut the source head's written directions out of the induction heads' keys. Run 2 edited all top 5 heads plus 5.0, and run 1 edited 4 of them plus 5.0, missing 7.2. The answers they named score 4 and 3 of 5 in `result.json`, because run 2 named the source head 4.11. `checks/audit_write_runs.py` finds no code in any write run that runs the model, and shows the two flags in `result.json` are print lines saying no forward pass was run.

| Edit | Repeat accuracy | Rise in text loss |
|---|---|---|
| Unedited | 0.987 | 0 |
| **Astra, run 1** | **0.180** | **0.013** |
| **Astra, run 2** | **0.011** | **0.013** |
| Simple edit, run 1's heads | 0.014 | 0.021 |
| Simple edit, run 2's heads | 0.002 | 0.022 |
| Simple edit, top 5 | 0.012 | 0.020 |
| Simple edit, next 5 | 0.985 | 0.000 |
| Simple edit, 16 random sets of 5, excluding the top ten | 0.972 to 0.987 | 0.019 to 0.268 |
| Simple edit, all 36 heads of layers 5 to 7 | 0.000 | 0.667 |
| Top 5 keys zeroed | 0.128 | 0.620 |
| Top 5 outputs zeroed | 0.295 | 1.066 |
| All of layers 5 to 7, outputs zeroed | 0.010 | 0.222 |
| Top 5 fixed at their mean output, which needs a forward pass | 0.016 | 0.004 |

The simple edit reduces each head's keys to what its constant query reads from them, the same projection Astra's run 1 used to protect that term, so it isn't independent of Astra. It needs only the head list. The outputs call it the head list edit. Rows come from `checks/outputs/edit_demo.txt`, `checks/outputs/edit_controls.txt` and `checks/outputs/write_baselines.txt`.

**The edit needs the list and the projection together.** On the top 5, zeroing keys costs 0.620 and the projection 0.020. The same edit does nothing to repeating on the next 5 or on random heads, including random heads in the right layers, so knowing the layers isn't enough for it. Those random sets exclude the top ten, and on short lists they range from 0.56 to 0.98, so lists are less specific than random tokens. Layers 5 to 7 can be hit without finding any head, by zeroing their outputs, at 0.222, about ten times the text cost of the short list. We measured what the list saves, not how hard it is to find, and a script with the known statistic also finds it. The top 5 is itself defined by attention on repeats, though Astra's own lists, chosen from weights alone, give 0.014 and 0.002.

**Other forms of edit.** Fixing the top 5 at their mean output does as well on repeating at less text cost, but needs running the model, which Astra couldn't do. Zeroing is a poor baseline. Run 1's weaker score comes from its edit and not its list, because the simple edit on the same heads reaches 0.014.

### What the edits do on plain English
The repeat score used random tokens and Astra ran nothing, so it never saw these lists. Prompts are fixed in `checks/edit_demo.py` before it runs, after earlier list checks, and every output is in `checks/outputs/edit_demo.txt`. A code review found our first name prompts started with the name, which is then a different token from the same name later on, so they didn't repeat at all. The rows below use the corrected prompts.

| On GPT-2 small | Unedited | Astra, run 1 | Astra, run 2 | Simple edit, run 2's heads | Top 5 outputs zeroed |
|---|---|---|---|---|---|
| Short lists of common nouns "a, b, c, a, b," right word | 0.95 | 0.14 | 0.05 | 0.05 | 0.40 |
| Five-word lists repeated, right word | 0.94 | 0.33 | 0.08 | 0.08 | 0.45 |
| Name again in a new sentence, right name | 0.25 | 0.44 | 0.25 | 0.23 | 0.17 |
| Name again in a second sentence, right name | 0.26 | 0.48 | 0.23 | 0.29 | 0.66 |
| Who lives where, right city | 0.15 | 0.13 | 0.36 | 0.44 | 0.01 |
| 30 plain facts, right on top | 15 | the same 15 | the same 15 | 14 | 10 |
| Top next token unchanged on stories | 1 | 0.95 | 0.95 | 0.96 | 0.52 |

**The edits weaken lists and random-token repeats, not every kind of repeat.** Repeated names in sentences don't fall, and run 1 gets them right more often. These heads aren't what carries that behaviour.

**On short lists of common words, the edited models usually repeat a word they just saw and rarely the next one.** Over 600 such lists they pick a listed word 91 to 100% of the time, the right one 4 to 14%, and mostly the last word again (`checks/outputs/edit_order.txt`). We wrote this check after seeing the examples. When the first word isn't at position 0, it comes back more often: "Words: apple, chair, river, apple, chair," gives "apple" in run 2, and the first word again rises to 37 and 44% (`checks/outputs/edit_controls.txt`). It doesn't hold everywhere (`checks/outputs/edit_limits.txt`):

| | Unedited | Astra, run 1 | Astra, run 2 | Simple edit, run 2's heads |
|---|---|---|---|---|
| 3-word lists, 4 other word pools, right word | 0.62 to 0.89 | 0.00 to 0.31 | 0.00 to 0.08 | 0.00 to 0.05 |
| Same lists, any listed word | 0.83 to 1.00 | 0.50 to 0.94 | 0.31 to 0.86 | 0.36 to 0.88 |
| Random tokens, top token anywhere in the first copy | 0.99 | 0.25 | 0.04 | 0.02 |
| Text loss rise, 256-token windows | 0 | 2.0% | 1.9% | 2.5% |

**Copying on real stories falls by less on run 1.** Over the same targets, a window repeated gets 0.91, and the same window after the text that really came before it gets 0.46, which is ordinary prediction. So the gain from repeating falls from 45 points to 29 in run 1 and 8 in run 2 (`checks/outputs/edit_controls.txt`). "The old man walked to the market and bought three fish. The old man walked to the market and bought" still continues "three fish." Run 2 also edited the heads' outputs, which alone do almost nothing (repeat 0.986, a local verifier check).

**Where the five heads look after the edit.** On random-token repeats they put 0.90 of their attention on the matching position, and after run 2's edit 0.01, while position 0 goes from 0.04 to 0.87. That is mostly their default: over the first copy, before anything repeats, they already put 0.92 there. So the edit leaves them at their default rather than sending them somewhere new. In run 1 four heads fall to 0.02 or below, and 7.2, which it didn't edit, stays at 0.58.

## A bigger model: reading held up better than writing
On scrambled GPT-2 medium, 355M weights and Astra's single run:
- **Reading:** the three sources it chose are that model's three strongest previous-token heads, 5.11, 4.13 and 6.7 (0.99, 0.96, 0.95, `checks/outputs/writer_statistic.txt`). It worked with several sources without being told there were several. Only 2 of its 5 targets are top-5 induction heads by attention.
- **Writing:** repeat accuracy 0.737 at text rise 0.135, which is not beyond deletion. Zeroing all of layers 5 to 7 gets 0.776 at 0.074. The simple edit does better on the same heads, 0.680 at 0.023, and also beats the same edit on the top 5 by attention, 0.901 at -0.033 (`checks/outputs/medium_head_list.txt`). These float32 scores move by up to 0.6 points with the thread count, so the check fixes it. It hit its $4.80 cap, but spent $6.43, about what small's run 2 spent, so budget doesn't obviously explain it. One run says little about size.

The known statistic also changes at this size. By best pairing, top-5 hits on attention score are 5, 2, 4 and 5 on GPT-2 small, medium, distilgpt2 and pythia-14m (`checks/outputs/other_models.txt`). Summing over strong writers gives 3, 4, 5, 5, a recipe written after seeing medium fail. No recipe wins everywhere.

Attention score alone misleads on medium. Zeroing the top 5 by attention costs 3.9 points of repeat accuracy, while best-pairing's 5 cost 41 to 43, against under 1 for same-layer heads. That doesn't mean the statistic sees what attention misses, since the earliest five high-attention heads cost about the same. Each head alone costs under 1 point, so the effect is joint, and zeroing them also raises text loss by 0.73 to 0.82 (`checks/outputs/write_baselines_medium.txt`). Removing their direct output costs 0.3 points, so the damage runs through later heads. How is unresolved (`checks/outputs/zero_order.txt`, `checks/outputs/medium_routes.txt`).

## How the mechanism holds up, on GPT-2 small
- **The earlier head is a previous-token head:** 4.11 is 98% on the previous token, the most of any head.
- **Several heads feed the keys together:** silencing the top 8 previous-token heads only for the five's keys takes their attention from 0.87 to 0.52, against 0.80 for random sets, and 4.11 alone gives 0.80. Through queries they don't move it (0.82, random 0.83). This holds with zero values (`checks/outputs/how_statistic_classifies.txt`), and with mean and resampled values in a local verifier check.
- **Later heads depend on the five through their queries:** 10.7 and 11.10 fall from about 0.5 to under 0.2 under all three ablations, with random sets flat, and both push the attended token down (`checks/outputs/signed_ladder.txt`).
- **4.11's space hurts most but isn't the only one that does:** removing its 64 directions from the keys hurts, but other heads' spaces do too, and much of the dramatic drop at low rank comes from position 0's huge norm (`checks/outputs/channel.txt`).
- **Attention to position 0 is a base rate:** 58 heads in layers 4 to 9 exceed 0.5 there. The five leaving it when a match appears is mostly softmax competition (`checks/outputs/first_position.txt`).

## What else can be read
Simple statistics on GPT-2 small, and with the token table where stated. These were chosen after trying many, with other models as the check.

| Kind of head | From the blocks alone | With the token table |
|---|---|---|
| Induction | 5 of 5 | no better |
| Duplicate | 4 of 5 | same hits with the table's rows shuffled, though that shuffle keeps each token's pairing, so it can't show the table is unused |
| Previous token | Top head ranks first on all 4 models. Most of the top 5 only on medium and distilgpt2 | not needed |
| Name mover | None | 4 of 5, through block 0's MLP |
| Negative heads like 10.7 and 11.10 | Not by value-output sign | First and second on GPT-2 small, and strongest ones first on medium and distilgpt2, Spearman about 0.8 on repeats, 0.43 to 0.54 on stories |
| S-inhibition | None | 1 of 5 at best |

This bears on [Ravfogel's question](https://x.com/ravfogel/status/2117642301442752692) of whether Astra reads only the first and last layers. Induction and previous-token heads read from middle blocks alone. For name movers and negative heads, the token table is what makes them readable to these statistics.

Asked openly to sort the heads, with no kind named (one run on GPT-2 small, `checks/outputs/astra_kinds.txt` and `runs/kind-open__live__20261104T043256Z`), Astra put the top 5 induction heads in one group, named real 4.11 as their source, and named layer 2's five highest previous-attention heads as a family. It also misplaced heads: 4.11 went into a 41-head group, and 10.7 and 11.10 weren't placed. Its "match and subtract" label doesn't hold up. Asked for duplicate heads, it got 4 of 5, the same as one line of code. Layer 2's family didn't generalize as a detector, in a local verifier check.

## What didn't work
- **Repeated names:** an earlier result here, that zeroing the five raises the right name, is withdrawn. Its prompts didn't repeat the name as a token. With corrected prompts, zeroing raises one template from 0.26 to 0.66 and lowers another from 0.25 to 0.17 (`checks/outputs/edit_demo.txt`). `checks/induction_examples.py`, `checks/signed_ladder.py` and `checks/why_names_rise.py` still use the old prompts.
- **Late heads don't carry medium's damage:** silencing medium's picks for later heads' queries cuts late attention 38% (`checks/outputs/medium_routes.txt`), but costs only 10 of the 41 points.

## Not checked
Astra's encrypted reasoning, so we can't tell whether it recognized GPT-2 or remembered the method. More than one or two runs per task. Other seeds for the medium runs. A test of whether Astra could read a model it couldn't have seen published. Verifiers' scripts and outputs are in a local ledger that isn't public yet, so results marked as local can't be rerun from this repo.
