"""Integration: the overview row must work for assets with an N/A historical score (crypto, leveraged)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dashboard as D  # noqa: E402
import engine as EN  # noqa: E402


class QuickNA(unittest.TestCase):
    def test_quick_and_analyze_with_na_score(self):
        D.DEMO = True
        old = EN.TABLE
        EN.TABLE = {"base_R": 0.04, "test_from": "2025-04-03", "stocks": 1, "from": "x", "to": "y", "buckets": {}}
        try:
            for s in ("BTC-USD", "NVDA"):
                q = D.quick(s)
                self.assertNotIn("error", q, q.get("error"))
                self.assertIn("score", q)
                a = D.analyze(s)
                self.assertIn("verdict", a["decision"])
        finally:
            EN.TABLE = old
            D.DEMO = False


if __name__ == "__main__":
    unittest.main()
