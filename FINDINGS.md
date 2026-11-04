# What later experiments found

These follow the [reading result](README.md) and ask how it works, what else can be read, whether reading can become writing, and what changes at a bigger size. Each result had its prediction written before it ran, and a separate agent reran or recomputed it, which often narrowed the claim. Several predictions failed, and those are marked. Astra's runs are in `runs/`, each one call log per turn. Totals are from one model and few runs, so treat them as leads rather than settled facts.

## Reading can become writing, on GPT-2 small
Asked to make the model stop repeating while keeping ordinary text, with Python, no forward pass and scrambled weights, Astra wrote edits to the weights and never saw a score. Both runs cut the source head's written directions out of the induction heads' keys.

| Edit | Repeat accuracy | Rise in text loss |
|---|---|---|
| Unedited | 0.987 | 0 |
| **Astra, run 1** | **0.180** | **0.013** (0.020 on a larger sample) |
| **Astra, run 2** | **0.011** | **0.013** (0.019 on a larger sample) |
| Zeroing the top 5 heads | 0.295 | 1.066 |
| Zeroing all of layers 5 to 7 | 0.010 | 0.222 |
| Reducing the same heads' keys to their query-bias term | 0.002 | 0.022 |

**This shows reading, not a cleverer edit.** The last row needs only the head list and matches Astra, so the value is in finding the right heads from scrambled weights. Zeroing was a poor baseline, because most of its text cost comes from losing each head's mean output. Copying on repeated real stories falls less, from 0.92 to about 0.53 in run 2 and 0.7 in run 1. Scored by `wsp/write_task.py`, with baselines in `checks/outputs/write_baselines.txt`.

## A bigger model: reading held up better than writing
On scrambled GPT-2 medium, 355M weights and Astra's single run:
- **Reading:** the three sources it chose are that model's three strongest previous-token heads, 5.11, 4.13 and 6.7 (0.99, 0.96, 0.95). It worked with several sources without being told there were several. Its targets are the set a short recipe also finds.
- **Writing:** repeat accuracy 0.737 at text rise 0.135, which is not beyond deletion and worse than a head-list edit on the same heads (0.680 at 0.023). That run hit its cost cap, so this isn't evidence of a limit of size.

The known statistic also changes at this size. By best pairing, top-5 hits on attention score are 5, 2, 4 and 5 on GPT-2 small, medium, distilgpt2 and pythia-14m (`checks/outputs/other_models.txt`). Summing over strong writers gives 3, 4, 5, 5, a recipe written after seeing medium fail. No recipe wins everywhere.

Attention score alone misleads on medium. Zeroing the top 5 by attention costs 3.9 points of repeat accuracy, while best-pairing's 5 cost 41 to 43, against under 1 for same-layer heads. That doesn't mean the statistic sees what attention misses, since the earliest five high-attention heads cost about the same. Each head alone costs under 1 point, so the effect is joint, and zeroing them also raises text loss by about 1.1. Removing their direct output costs 0.3 points, so the damage runs through later heads. How is unresolved (`checks/outputs/zero_order.txt`, `checks/outputs/medium_routes.txt`).

## How the mechanism holds up, on GPT-2 small
- **The earlier head is a previous-token head:** 4.11 is 98% on the previous token, the most of any head.
- **Several heads feed the keys together:** silencing the top 8 previous-token heads only for the five's keys takes their attention from 0.87 to 0.52, against 0.80 for random sets, and 4.11 alone gives 0.80. Through queries they don't move it (0.82, random 0.83). This holds with zero, mean and resampled values (`checks/outputs/how_statistic_classifies.txt`).
- **Later heads depend on the five through their queries:** 10.7 and 11.10 fall from about 0.5 to under 0.2 under all three ablations, with random sets flat, and both push the attended token down (`checks/outputs/signed_ladder.txt`).
- **4.11's own space isn't special:** removing its 64 directions from the keys hurts, but other heads' spaces do too, and much of the dramatic drop at low rank comes from position 0's huge norm (`checks/outputs/channel.txt`).
- **Attention to position 0 is a base rate:** 58 heads in layers 4 to 9 exceed 0.5 there. The five leaving it when a match appears is mostly softmax competition (`checks/outputs/first_position.txt`).

## What else can be read
Simple statistics on GPT-2 small, and with the token table where stated. These were chosen after trying many, with other models as the check.

| Kind of head | From the blocks alone | With the token table |
|---|---|---|
| Induction | 5 of 5 | no better |
| Duplicate | 4 of 5 | same statistic works on random tables, so it isn't reading them |
| Previous token | Top head ranks first on all 4 models. Most of the top 5 only on medium and distilgpt2 | not needed |
| Name mover | At most 1 of 5 | 4 of 5, through block 0's MLP |
| Negative heads like 10.7 and 11.10 | Not by value-output sign | First and second on GPT-2 small, and strongest ones first on medium and distilgpt2, Spearman about 0.8 |
| S-inhibition | None | 1 of 5 at best |

This bears on [Ravfogel's question](https://x.com/ravfogel/status/2117642301442752692) of whether Astra reads only the first and last layers. Induction and previous-token heads read from middle blocks alone. For name movers and negative heads, the token table is what makes them readable to these statistics.

Asked openly to sort the heads, with no kind named (one run on GPT-2 small, `checks/outputs/astra_kinds.txt`), Astra put the top 5 induction heads in one group, named real 4.11 as their source, and named layer 2's five highest previous-attention heads as a family. It also misplaced heads: 4.11 went into a 41-head group, and 10.7 and 11.10 weren't placed. Its "match and subtract" label doesn't hold up. Asked for duplicate heads, it got 4 of 5, the same as one line of code. Layer 2's family didn't generalize as a detector.

## What didn't work
- **Repeated names:** zeroing the five raises the right name in 19 of 22 prompt sets, and with mean or resampled values in 0 of 22, so the rise is mostly an artifact of zeroing.
- **Late heads don't carry medium's damage:** silencing medium's picks for later heads' queries cuts late attention 40%, but costs only 10 of the 41 points.

## Not checked
Astra's encrypted reasoning, so we can't tell whether it recognized GPT-2 or remembered the method. More than one or two runs per task. Other seeds for the medium runs. A test of whether Astra could read a model it couldn't have seen published. Verifiers' scripts and outputs are in a local ledger that isn't public yet.
