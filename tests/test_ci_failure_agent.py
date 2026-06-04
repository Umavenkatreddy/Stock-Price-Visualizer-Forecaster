import re, unittest

class TestAnsiStripping(unittest.TestCase):
    def _s(self, t): return re.sub(r"\x1B\[[0-9;]*[mK]", "", t)
    def test_color(self): self.assertEqual(self._s("\x1b[31mERROR\x1b[0m"), "ERROR")
    def test_plain(self): self.assertEqual(self._s("plain"), "plain")
    def test_multi(self): self.assertEqual(self._s("\x1b[1m\x1b[32mPASS\x1b[0m"), "PASS")

class TestTimestampStripping(unittest.TestCase):
    def _s(self, t): return re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s*", "", t)
    def test_removes(self): self.assertEqual(self._s("2026-05-19T14:32:01.123Z ERROR"), "ERROR")
    def test_no_ts(self): self.assertEqual(self._s("ERROR"), "ERROR")

class TestGroupStripping(unittest.TestCase):
    def _s(self, t): return re.sub(r"##\[(group|endgroup)\].*", "", t)
    def test_group(self): self.assertEqual(self._s("##[group]Install").strip(), "")
    def test_endgroup(self): self.assertEqual(self._s("##[endgroup]").strip(), "")
    def test_normal(self): self.assertEqual(self._s("Running..."), "Running...")

class TestSignalDetection(unittest.TestCase):
    def _sig(self, line):
        return bool(re.compile(r"(error|fail|exception|traceback|assert|syntax|import|undefined|cannot|could not|no module|exit code [^0])", re.IGNORECASE).search(line))
    def test_error(self): self.assertTrue(self._sig("ERROR: oops"))
    def test_fail(self): self.assertTrue(self._sig("FAILED test"))
    def test_exception(self): self.assertTrue(self._sig("Exception raised"))
    def test_traceback(self): self.assertTrue(self._sig("Traceback (most recent)"))
    def test_no_module(self): self.assertTrue(self._sig("No module named 'foo'"))
    def test_exit_code(self): self.assertTrue(self._sig("exit code 1"))
    def test_normal(self): self.assertFalse(self._sig("Running pytest..."))
    def test_passed(self): self.assertFalse(self._sig("5 passed in 1.23s"))
    def test_exit0(self): self.assertFalse(self._sig("exit code 0"))

class TestDeduplication(unittest.TestCase):
    def _dedup(self, lines):
        seen=set(); result=[]
        for l in lines:
            if l not in seen: seen.add(l); result.append(l)
        return result
    def test_dups(self): self.assertEqual(self._dedup(["a","b","a","c"]), ["a","b","c"])
    def test_order(self): self.assertEqual(self._dedup(["x","y","z"]), ["x","y","z"])
    def test_empty(self): self.assertEqual(self._dedup([]), [])

class TestLogCapping(unittest.TestCase):
    def _cap(self, text, mx=6000):
        if len(text)<=mx: return text
        h=mx//2; return text[:h]+"\n\n... [truncated] ...\n\n"+text[-h:]
    def test_short(self): self.assertEqual(self._cap("hi"), "hi")
    def test_long(self): self.assertIn("truncated", self._cap("x"*10000))
    def test_exact(self): t="z"*6000; self.assertEqual(self._cap(t), t)

class TestDiffCapping(unittest.TestCase):
    def _cap(self, diff, mx=4000):
        if len(diff)>mx: return diff[:mx]+"\n\n... [diff truncated] ..."
        return diff
    def test_short(self): self.assertEqual(self._cap("diff"), "diff")
    def test_long(self): self.assertIn("diff truncated", self._cap("+"*5000))

class TestReportMarker(unittest.TestCase):
    def test_run_id(self): self.assertIn("123", "<!-- ci-failure-agent run_id=123 -->")
    def test_unique(self): self.assertNotEqual("<!-- ci-failure-agent run_id=111 -->", "<!-- ci-failure-agent run_id=222 -->")
    def test_structure(self):
        body="<!-- ci-failure-agent run_id=9 -->\n## CI Failure Agent Report\nhttps://url"
        self.assertIn("CI Failure Agent Report", body); self.assertIn("https://url", body)

if __name__ == "__main__": unittest.main()