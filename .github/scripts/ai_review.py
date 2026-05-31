"""
AI PR Review — runs in GitHub Actions, calls GitHub Models (free), and posts
a single Claude-style review comment on the PR.

Auth model:
  - GITHUB_TOKEN (built-in workflow secret) is used both to:
      1. Call the GitHub Models inference endpoint (with `models: read`)
      2. Post the review comment back to the PR (`pull-requests: write`)

No Anthropic key, no OpenAI account, no SAP AI Core, no hai proxy. Free.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from openai import OpenAI
from github import Github, Auth


# ---------- 1. Validate environment ----------------------------------------

REQUIRED = ("GITHUB_TOKEN", "GITHUB_REPOSITORY", "PR_NUMBER")
missing = [k for k in REQUIRED if not os.getenv(k)]
if missing:
    print(f"ERROR: missing env vars: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)

token = os.environ["GITHUB_TOKEN"]
repo_full = os.environ["GITHUB_REPOSITORY"]
pr_number = int(os.environ["PR_NUMBER"])
api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
model = os.environ.get("AI_REVIEW_MODEL", "openai/gpt-4o-mini")


# ---------- 2. Read the diff ------------------------------------------------

DIFF_FILE = Path("pr.diff")
if not DIFF_FILE.exists() or DIFF_FILE.stat().st_size == 0:
    print("No diff to review — exiting cleanly.")
    sys.exit(0)

diff_text = DIFF_FILE.read_text(encoding="utf-8", errors="replace")
print(f"Reviewing diff: {len(diff_text)} chars, model={model}")


# ---------- 3. Build the prompt --------------------------------------------

PR_TITLE = os.getenv("PR_TITLE", "(no title)")
PR_BODY = os.getenv("PR_BODY") or "(no description)"

SYSTEM_PROMPT = """You are a senior Python code reviewer for a Streamlit/Dash + LSTM/SVR
stock-forecasting project. Review the supplied unified diff and produce a
concise, actionable review.

Output format (Markdown):

### 🤖 AI Review Summary
1–3 sentences on the overall change.

### ✅ Looks good
- bullet points (or "None" if nothing notable)

### ⚠️ Issues / Risks
For each issue:
- **<file>:<line-range>** — <short title>
  - Why it matters: ...
  - Suggested fix:
    ```python
    <minimal patch or pseudocode>
    ```

### 🧪 Test coverage
Note any new behavior NOT covered by tests under `tests/`. Write "None" if no gaps.

### 🔒 Security / secrets
Flag hard-coded credentials, API keys, unsafe `eval`, command/SQL injection.
Write "None" if clean.

Rules:
- Be specific — quote file paths and line numbers from the diff hunk headers.
- Do NOT repeat the diff back.
- Keep the whole review under ~400 lines."""

USER_PROMPT = f"""PR title: {PR_TITLE}

PR description:
{PR_BODY}

Unified diff (truncated if larger than 150 KB):

```diff
{diff_text}
```
"""


# ---------- 4. Call GitHub Models ------------------------------------------

client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=token,
)

try:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    review_md = (resp.choices[0].message.content or "").strip()
    if not review_md:
        review_md = "_AI review produced no output._"
except Exception as exc:
    # Soft fail — still post a comment so the PR author can see what went wrong.
    print(f"ERROR calling GitHub Models ({model}): {exc}", file=sys.stderr)
    review_md = (
        f"⚠️ AI review failed to run.\n\n"
        f"- Model: `{model}`\n"
        f"- Error: `{exc}`\n\n"
        "Likely causes: GitHub Models not enabled on this org, the workflow "
        "lacks `models: read` permission, or a temporary outage."
    )


# ---------- 5. Post / update the PR comment --------------------------------

gh = Github(auth=Auth.Token(token), base_url=api_url)
repo = gh.get_repo(repo_full)
pr = repo.get_pull(pr_number)

MARKER = "<!-- ai-pr-review-bot -->"
body = (
    f"{MARKER}\n{review_md}\n\n"
    f"<sub>🤖 Model: `{model}` · Auto-updates on every push.</sub>"
)

# Idempotent: update the existing bot comment if present, else create one.
existing = None
for c in pr.get_issue_comments():
    if c.body and MARKER in c.body:
        existing = c
        break

if existing:
    existing.edit(body)
    print(f"Updated existing AI review comment id={existing.id}")
else:
    new_comment = pr.create_issue_comment(body)
    print(f"Posted new AI review comment id={new_comment.id}")