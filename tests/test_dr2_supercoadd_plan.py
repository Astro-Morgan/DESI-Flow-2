import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from DESIFlow.data import dr2_cache as D


def test_sc_plan_lists_members_of_multi_row_targets_only():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "index"))
        tid = np.array([7, 3, 7, 5, 3, 3, 9], np.int64)                            # tgt rows 0..6
        np.save(os.path.join(d, "index", "targetid.npy"), np.r_[tid, 100])         # a sky row that is not a tgt row
        np.save(os.path.join(d, "index", "tgt_rows.npy"), np.arange(7))
        mem, start, count, t = D.sc_plan(d)
    assert t.tolist() == [3, 7] and count.tolist() == [3, 2] and start.tolist() == [0, 3]
    assert mem[0:3].tolist() == [1, 4, 5] and mem[3:5].tolist() == [0, 2]
    return "sc_plan: targets 3 and 7 are multi-row, member rows listed in row order, single-row targets 5 and 9 dropped"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                print(f"PASS {name}: {fn()}")
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
