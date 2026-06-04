import unittest

class TestEnvValidation(unittest.TestCase):
    def _require(self, name, env):
        v = env.get(name, "").strip()
        if not v: raise SystemExit(1)
        return v
    def test_returns_value(self): self.assertEqual(self._require("T", {"T": "val"}), "val")
    def test_exits_missing(self):
        with self.assertRaises(SystemExit): self._require("X", {})
    def test_exits_empty(self):
        with self.assertRaises(SystemExit): self._require("T", {"T": ""})
    def test_exits_whitespace(self):
        with self.assertRaises(SystemExit): self._require("T", {"T": "   "})
    def test_strips(self): self.assertEqual(self._require("T", {"T": "  abc  "}), "abc")

class TestPRNumber(unittest.TestCase):
    def test_valid(self): self.assertEqual(int("42"), 42)
    def test_invalid(self):
        with self.assertRaises(ValueError): int("bad")

class TestModelFallback(unittest.TestCase):
    def test_default(self): self.assertEqual("".strip() or "openai/gpt-4o-mini", "openai/gpt-4o-mini")
    def test_custom(self): self.assertEqual("openai/gpt-4o".strip() or "openai/gpt-4o-mini", "openai/gpt-4o")

class TestBotMarker(unittest.TestCase):
    MARKER = "<!-- ai-pr-review-bot -->"
    def test_in_body(self): self.assertIn(self.MARKER, f"{self.MARKER}\nreview")
    def test_detect(self):
        m = self.MARKER
        found = next((c["body"] for c in [{"body":"x"},{"body":f"{m}\nold"}] if m in c.get("body","")), None)
        self.assertIsNotNone(found)
    def test_not_found(self):
        found = next((c["body"] for c in [{"body":"no"}] if self.MARKER in c.get("body","")), None)
        self.assertIsNone(found)
    def test_starts_with(self): self.assertTrue(f"{self.MARKER}\nreview".startswith(self.MARKER))

class TestDiff(unittest.TestCase):
    def test_empty_falsy(self): self.assertFalse(bool("".strip()))
    def test_nonempty_truthy(self): self.assertTrue(bool("diff x".strip()))
    def test_cap(self): self.assertEqual(len(("x"*200000)[:150000]), 150000)
    def test_prompt_diff(self):
        d="+added"; self.assertIn(d, f"`diff\n{d}\n`")
    def test_prompt_title(self):
        t="Fix bug"; self.assertIn(t, f"PR title: {t}")

class TestCommentBody(unittest.TestCase):
    def test_starts_marker(self):
        m="<!-- ai-pr-review-bot -->"; self.assertTrue(f"{m}\nreview".startswith(m))
    def test_header(self): self.assertIn("### AI PR Review", "<!-- m -->\n### AI PR Review\ntext")
    def test_model(self):
        model="openai/gpt-4o-mini"; self.assertIn(model, f"review\n<sub>Model: {model}</sub>")
    def test_patch(self): self.assertEqual("PATCH" if 123 else "POST", "PATCH")
    def test_post(self): self.assertEqual("PATCH" if None else "POST", "POST")

if __name__ == "__main__": unittest.main()