import unittest
from collections import deque
from pathlib import Path
from typing import Dict, List


def _load_deque_collection():
    source = Path(
        "/config/workspace/sglang/python/sglang/srt/eplb/expert_distribution.py"
    ).read_text()

    start = source.index("class _DequeCollection:")
    end = source.index("\n\nclass _DetailAccumulator", start)
    namespace = {"deque": deque, "Dict": Dict, "List": List}
    exec(source[start:end], namespace)
    return namespace["_DequeCollection"]


class TestDequeCollection(unittest.TestCase):
    def test_mean_skips_empty_windows_after_clear(self):
        deque_collection_cls = _load_deque_collection()
        history = deque_collection_cls(maxlens=[10, 100])
        history.append(1.0)
        history.append(3.0)

        self.assertEqual(history.mean(), {10: 2.0, 100: 2.0})

        history.clear()

        self.assertEqual(history.mean(), {})


if __name__ == "__main__":
    unittest.main()
