"""Cross-process equivalence check for the blocklane shared tier: a spawned second process
(with its own _mem, like the real workers) reads a block another process published — the same
shape as tools/benchmark_range_cache.py but for the shared-memory lane."""
import multiprocessing as mp
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import blocklane                              # noqa: E402


def _worker(path, q):
    import blocklane
    sh = blocklane.store_for(path)
    got = sh.read(blocklane.key_of(path), 7)
    q.put(got)


class SharedTierTest(unittest.TestCase):
    def test_a_spawned_process_reads_the_block(self):
        with tempfile.TemporaryDirectory() as d:
            sh = blocklane.store_for(d)
            k = blocklane.key_of(d)
            sh.write(k, 7, b"A" * 5000)
            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            p = ctx.Process(target=_worker, args=(d, q))
            p.start()
            p.join(15)
            self.assertEqual(q.get(timeout=10), b"A" * 5000)

    def test_degrades_when_off(self):
        import os as _os
        _os.environ["BNGE_SHARED_MEM"] = "off"
        try:
            sh = blocklane.store_for(tempfile.gettempdir())
            self.assertIsNone(sh.read(1234, 0))
        finally:
            del _os.environ["BNGE_SHARED_MEM"]


if __name__ == "__main__":
    unittest.main()
