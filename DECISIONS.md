# Decisions: what Alex said and what we chose

Current design: [METHOD.md](METHOD.md). Evidence: [CONTEXT.md](CONTEXT.md). Quotes are Alex's words as typed, in order.

## Where it started
> "how does this scale with size, if there is some intrinsic scaling laws in LLMs selecting the weights of other LLMs. Do you think that we can do a small study about it consuming few compute and tokens?"

> "astra tokens we have like 2 week plans of 100 usd of codex"

> "but beware of" [Karvonen's reply]

## Decisions
| # | Alex said | Decision | Why |
|---|---|---|---|
| 1 | "why use a game? I wanted to write LLMs weights" | Target LLMs, not game policies | Games were only a cheap proxy with a reference score |
| 2 | "the problem with that would be the benchmarking of the models, on which datasets and tasks" | Tasks a small written model can score on | MATH-500 and IFEval would likely sit near zero at every size |
| 3 | "should probably do something more than reasoning, though one should start small" | Tasks from different abilities, at one size first | Their eight benchmarks may mostly measure whether the model finishes |
| 4 | "first we really need to setup all the method and then we can run" | Nothing runs until the method is settled | |
| 5 | "that method feels like cheating, right? we would probably like to really measure the ability of the model on its own setting the weights" | No Python, forward passes, feedback or retries | With code, we would measure coding as much as setting weights |
| 6 | "Allow ranges, no arithmetic" | Typed values and ranges, no `*=`, `+=`, `-=` | A range is notation, while the operators start computing for Astra |
| 7 | "i would keep layer norm because we will grab a a ready made architecture and set the weights" | Use ready-made architectures with their normalization | Replaces a custom no-norm network we had proposed |
| 8 | "i would look for popular reknown models" | Well-known models, not TinyStories-1M | TinyStories-1M was our suggestion, and it stays only as a dataset |
| 9 | "we dont have to replicate everythin like them" | Not a replication | One Breakout run stays, as a harness check |
| 10 | "we want to try Astra as setting the weights on models and see it it works or if it doesnt, to do the proper experiment these authors claim" | The goal is a plain test of whether it works | Scaling comes after a positive result |
| 11 | "we can first do it without tools and then maybe with tool calls, like letting it run python program, but that is another study" | No tools now, Python is a later study | |
| 12 | "gpt 2 seems like a better start" | GPT-2 small first | |
| 13 | "it is also interesting to compare in teh same model they did, qwen3.5 0.8B" | Qwen3.5-0.8B on the same tasks, after GPT-2 | See open question 1 |
| 14 | On the Gemma failure: "is important" | Report reading and writing separately, with a later reading study | It is their only test of reading at 268M or more without running the model |
| 15 | "Can Astra (or LMs) really understand the weights of another LLM and write them in an optimal way? (or maybe that is too much and we shoukd even narrow it)" | Narrowed: can it write GPT-2 weights for a task, and predict what they do? | "Optimal" cannot be tested, and "understand" needed a concrete test |
| 16 | "im scared that GPT2 weights might even be in training data, but its a good starting point" | Shuffle token IDs, do not name the model, compare typed values with real GPT-2 | Copying can be ruled out, general knowledge from papers cannot |
| 17 | "the problem with the capital of france is that it is the most common frace here, so we can try it with several" | Four phrasings, two hidden, with well-known and obscure capitals scored separately | A single phrasing lets a network read one position with no attention |
| 18 | "the gpt 2 can probably be tried now locally, for astra we can call it through codex. Log all the calls" | Score GPT-2 on the Mac, call Astra through sealed Codex, save every call in full | |
| 19 | "we dont want to train networks ourselves, but let the model call to astra set the weights or understand the weights" | Reading test on ready-made GPT-2 small | Written networks turned out to be small circuits in an empty container, so a family sweep would be flat |
| 20 | "we can let the own model decide how to read them", then "Viewer only" | A read-only viewer with a value budget, no computation | Whole-model reading does not fit in context, and fixed slices would choose for it |
| 21 | "Yes, small only" | GPT-2 small only for now | A size sweep waits for a positive result and a new decision |
| 22 | "Yes, hinted read" | Hinted read after the first read scored 0 of 5 | Separates not knowing what to compute from being unable to compute it |
| 23 | "so run it" | Second unhinted read as well | So the headline does not rest on one run |
| 24 | "Yes, public repo" | Public repo, leaving out third-party text | The post links to the code and logs, and GPL MinAtar source, the post's answer and Codex's catalog are not ours to republish |

## Things we corrected along the way
- **Beating chance was too easy.** On capitals, one right answer out of 190 would have passed. Each task must now beat the best trivial strategy.
- **The token table is not as limiting as first stated.** We assumed every token needs all its numbers typed, but a sparse code needs only one or two.
- **About 800 values is not a hard limit.** It is what Astra chose to write in those runs.
- **Dropped N^0.36.** That guess came from two runs on different tasks, one of which used activations.

## Open questions
1. **Qwen timing.** The narrowed design runs Qwen only if GPT-2 shows signal. Alex wanted that comparison and has not objected to the gate, so this is our assumption and he can still veto it.
2. **Failures.** Successes get repeated 3 times. Failures are reported as "failed in one attempt", and repeated only if the budget allows.
3. **Scope of Predict.** It reads sparse written networks, so it says nothing about dense trained weights. That stays for the later reading study.
