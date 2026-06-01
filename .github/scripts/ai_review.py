"""
AI PR Review — calls GitHub Models (free) and posts a PR comment.

Auth: GITHUB_TOKEN (built-in workflow secret).
  - POST https://models.github.ai/inference  (with models:read permission)
  - POST https://api.github.com/repos/.../issues/.../comments  (pull-requests:write)

Dependencies: openai, requests  (both pinned in the workflow pip install step)
No PyGithub, no Anthropic SDK, no paid keys.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import requests as http

# ---------- startup diagnostics (visible in Actions log) --------------------
print(f"Python {sys.version}")
print(f"Script: {__file__}")
REQUIRED_VARS = ("GITHUB_TOKEN", "GITHUB_REPOSITORY", "PR_NUMBER")
for v in REQUIRED_VARS:
    val = os.getenv(v, "")
    safe = f"[set, len={len(val)}]" if val else "[MISSING]"
    print(f"  {v}: {safe}")
print(f"  GITHUB_API_URL: {os.getenv('GITHUB_API_URL', '(not set)')}")
print(f"  AI_REVIEW_MODEL: {os.getenv('AI_REVIEW_MODEL', '(not set)')}")


# ---------- 1. Collect env vars -------------------------------------------

def _require(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        print(f"FATAL: env var {name!r} is empty or missing.", file=sys.stderr)
        sys.exit(1)
    return v


token       = _require("GITHUB_TOKEN")
repo_full   = _require("GITHUB_REPOSITORY")    # e.g. Umavenkatreddy/Stock-Price-Visualizer-Forecaster
pr_number   = _require("PR_NUMBER")            # we keep as str, convert later
api_base    = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")
model       = os.getenv("AI_REVIEW_MODEL", "").strip() or "openai/gpt-4o-mini"

# Convert pr_number to int after verification
try:
    pr_number_int = int(pr_number)
except ValueError:
    print(f"FATAL: PR_NUMBER={pr_number!r} is not an integer.", file=sys.stderr)
    sys.exit(1)


# ---------- 2. Read the diff ------------------------------------------------

DIFF_FILE = Path("pr.diff")
if not DIFF_FILE.exists() or DIFF_FILE.stat().st_size == 0:
    print("No diff to review — empty or missing pr.diff. Exiting cleanly.")
    sys.exit(0)

diff_text = DIFF_FILE.read_text(encoding="utf-8", errors="replace")
print(f"diff size: {len(diff_text)} chars")


# ---------- 3. Build the prompt --------------------------------------------

PR_TITLE = os.getenv("PR_TITLE", "(no title)").strip()
PR_BODY  = (os.getenv("PR_BODY") or "(no description)").strip()

SYSTEM_PROMPT = """You are a senior Python code reviewer for a Streamlit/Dash + LSTM/SVR
stock-forecasting project. Review the supplied unified diff and produce a
concise, actionable review.

Output format (Markdown):

### Summary
1-3 sentences on the overall change.

### Looks good
- bullet points (or "None")

### Issues / Risks
For each issue:
- **<file>:<line>** - <short title>
  - Why it matters: ...
  - Suggested fix: `<minimal fix>`

### Test coverage
Note new behavior NOT covered by tests. Write "None" if no gaps.

### Security
Flag hard-coded credentials, API keys, unsafe eval, command injection.
Write "None" if clean.

Rules: be specific, quote file+line numbers, do NOT repeat the diff."""

USER_PROMPT = (
    f"PR title: {PR_TITLE}\n\n"
    f"PR description:\n{PR_BODY}\n\n"
    f"Unified diff:\n\n```diff\n{diff_text}\n```"
)


# ---------- 4. Call GitHub Models via openai SDK ---------------------------

try:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=token,
    )
    print(f"Calling model={model} via https://models.github.ai/inference …")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT},
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    review_md = (resp.choices[0].message.content or "").strip()
    if not review_md:
        review_md = "_AI review produced no output._"
    print(f"Model response: {len(review_md)} chars")

except Exception as exc:
    import traceback
    traceback.print_exc()
    review_md = (
        f"## AI Review — model call failed\n\n"
        f"- Model: `{model}`\n"
        f"- Error: `{exc}`\n\n"
        f"Possible causes: GitHub Models not enabled on this organisation, "
        f"rate-limit, or temporary outage. "
        f"Check the [workflow run]({api_base.replace('api.', '')}) for details."
    )


# ---------- 5. Post / update PR comment via REST ---------------------------

MARKER = "<!-- ai-pr-review-bot -->"
body = (
    f"{MARKER}\n"
    f"### AI PR Review\n\n"
    f"{review_md}\n\n"
    f"<sub>Model: `{model}` · Updates on every push.</sub>"
)

gh_headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# List existing PR comments to find a previous bot comment.
comments_url = f"{api_base}/repos/{repo_full}/issues/{pr_number_int}/comments"
existing_id  = None
page = 1
while True:
    r = http.get(comments_url, headers=gh_headers, params={"per_page": 100, "page": page})
    if r.status_code != 200:
        print(f"WARNING: could not list comments ({r.status_code}): {r.text[:200]}")
        break
    comments = r.json()
    if not comments:
        break
    for c in comments:
        if MARKER in (c.get("body") or ""):
            existing_id = c["id"]
            break
    if existing_id or len(comments) < 100:
        break
    page += 1

# Update or create.
if existing_id:
    r = http.patch(
        f"{api_base}/repos/{repo_full}/issues/comments/{existing_id}",
        headers=gh_headers,
        json={"body": body},
    )
    if r.status_code == 200:
        print(f"Updated existing AI review comment id={existing_id}")
    else:
        print(f"ERROR updating comment: {r.status_code} {r.text[:300]}", file=sys.stderr)
        sys.exit(1)
else:
    r = http.post(
        comments_url,
        headers=gh_headers,
        json={"body": body},
    )
    if r.status_code == 201:
        print(f"Posted new AI review comment: {r.json()['html_url']}")
    else:
        print(f"ERROR posting comment: {r.status_code} {r.text[:300]}", file=sys.stderr)
        sys.exit(1)