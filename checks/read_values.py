"""Counts how many query and key values each reading run looked at, from its saved transcript."""
import glob
import json
import re

import numpy as np

from wsp.disguise import WIDTH
from wsp.edits import _parse_index

QKV_SHAPE = (WIDTH, 3 * WIDTH)
_STATEMENT = re.compile(r"^L\d+\.qkv\.W\[(?P<index>[^\]]*)\] =$", re.MULTILINE)

columns = np.tile(np.arange(QKV_SHAPE[1]), (QKV_SHAPE[0], 1))
for run in sorted(glob.glob("runs/reading-*__live__*")):
    query_key = 0
    for line in open(f"{run}/transcript.jsonl"):
        for match in _STATEMENT.finditer(json.loads(line).get("viewer") or ""):
            query_key += int((np.asarray(columns[_parse_index(match["index"], QKV_SHAPE)]) < 2 * WIDTH).sum())
    print(f"{run}: {query_key} query and key values")
