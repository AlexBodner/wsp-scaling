# What do the first and last tables add for each kind of head?

Question: Q001. Which kinds of head can be read from the blocks alone, and which need the token, position or final tables?
Rests on: checks/outputs/head_types.txt: blocks-only statistics find induction 5, duplicate 4, previous 0, S-inhibition 0, name mover at most 1. Ravfogel's public question.
Mechanism: previous-token heads are defined by position, so they should need P. Name movers copy a token to the output, so a copying score needs E and F. Duplicate heads match a token to itself, which QK^T can show without E because residual directions are near isotropic. Induction needs only QK^T against an earlier VO.
Prediction: previous goes from 0 to at least 3 of 5 with P. Name mover goes from at most 1 to at least 3 with E and F. Duplicate stays at 3 or more with or without E. Induction stays at 5 with blocks alone and gains nothing. S-inhibition stays at 1 or less with every table, since it writes a signal rather than a token.
Control: the same statistics with the tables scrambled (token rows permuted against the unembedding, or position rows permuted). They should fall to about chance, 0.17 of 5, unless the statistic isn't really using the table.
If it comes out as predicted: the answer to Ravfogel's question is that it depends on the kind, and the boundary tables matter for previous and name mover but not induction. Use this to choose which kinds Astra is tested on and with which tables.
If it comes out otherwise: if scrambled tables score as well, that statistic isn't reading the table. If previous stays at 0 with P, position heads aren't linearly readable from P and QK^T, which would be a real limit on reading.
Output: checks/tables_by_kind.py writing checks/outputs/tables_by_kind.txt
Cost: free, under 30 minutes.
