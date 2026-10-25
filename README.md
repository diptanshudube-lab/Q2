# Invoice Action Agent (A2A, HTTP+JSON)

A small web service that reads invoice "claim batches", picks one safe action per invoice with Claude, shows the evidence it used, and only records a receipt after the grader accepts the proposal. It is built to run on Render.

## The easy version (how it works, step by step)

1. **Someone sends a batch.** A batch is a box of invoice packages. Each package has documents with little reference tags in square brackets, like `[INV-A]`.
2. **The agent asks Claude to read the packages.** Claude picks one of five actions for each invoice: pay it, ask for approval, hold it, reject it as a duplicate, or open an exception. Claude must copy the exact reference tags it relied on.
3. **The agent checks Claude's answer.** Every answer must use real tags from the documents, real amounts, and real invoice numbers. Bad answers are sent back for a fix, once more.
4. **The agent saves a proposal and waits.** Nothing is paid at this point. It says "input required" and waits for the grader.
5. **The grader replies with results.** For each invoice it says ACCEPTED or REJECTED, with a receipt code.
6. **The agent records receipts for accepted items only.** Then the task is done.
7. **Safety rules.** Asking twice with the same message gives the same answer and never runs Claude again. Each user (each Bearer token) only sees their own tasks. Saved state lives in a database file, so a restart does not lose it.

## What is in this folder

| File | What it does |
|---|---|
| `app/main.py` | The web routes, Bearer login, version and media-type checks |
| `app/protocol.py` | A2A rules, the Agent Card, and input validation |
| `app/service.py` | Task lifecycle: send, results, cancel, read, list, duplicates, races |
| `app/decider.py` | The Claude call and the checks on every answer |
| `app/store.py` | SQLite database (tasks, message IDs, decision cache, daily call count) |
| `app/config.py` | Reads settings from environment variables |
| `tests/` | 37 automated tests. They use a fake decider, so they need no API key |
| `scripts/live_check.py` | Sends one small fake batch to your deployed agent with real Claude, then accepts it |
| `render.yaml` | Render setup (service, start command, disk, environment variables) |
| `requirements.txt` | Pinned Python packages |

## Step 1: Run the tests on your computer

You need Python 3.11 or newer (3.13 is what this was tested on).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest -q
```

Expected output: `37 passed`.

## Step 2: Get an Anthropic API key

Create a key at the Anthropic Console. Keep it private. You will paste it into Render in Step 4, never into a file or a chat.

## Step 3: Put the code on GitHub

Render deploys from a Git repository. Create an empty GitHub repository, then from this folder:

```bash
git init
git add .
git commit -m "Invoice action agent"
git branch -M main
git remote add origin https://github.com/YOUR-NAME/YOUR-REPO.git
git push -u origin main
```

`.gitignore` already keeps the database, virtual environments and secrets out of Git.

## Step 4: Deploy on Render

1. In Render, choose **New → Blueprint** and select your repository. Render reads `render.yaml`.
2. Render will ask for the values marked `sync: false`:
   - `ANTHROPIC_API_KEY`: your key from Step 2.
   - `PUBLIC_BASE_URL`: leave it empty for the first deploy (see Step 5).
   - `A2A_ALLOWED_TOKENS`: optional. See the security note below.
3. Click **Apply**. The first build takes a few minutes.

**Why a paid plan?** The file uses `plan: starter` because Render disks (where the database lives) are only available on paid plans. Without a disk, every restart would forget tasks, and retries could repeat work.

## Step 5: Set the exact base URL

1. After the deploy, copy your service URL, for example `https://invoice-action-agent.onrender.com`.
2. In the Render dashboard, open **Environment** and set `PUBLIC_BASE_URL` to that URL plus `/a2a/`, for example:

   `https://invoice-action-agent.onrender.com/a2a/`

   It must match the base URL you submit to the grader **exactly**, including the trailing slash if you use one. The Agent Card publishes this value.
3. Save. Render redeploys.

Check it works (this does not use any grader check):

```bash
curl https://invoice-action-agent.onrender.com/.well-known/agent-card.json
```

You should see the card with `"url": "https://invoice-action-agent.onrender.com/a2a/"`.

## Step 6: Run a live check before you use the grader

The page only leaves you **one** check, and it is reserved for Save. So test the real Claude path yourself first. This sends two fake packages, prints Claude's decisions and the timings, then accepts them. It costs a few cents of API credit.

```bash
python scripts/live_check.py --base https://invoice-action-agent.onrender.com/a2a/ --token my-test-token
```

Look at:
- Each request should take well under 45 seconds (the script prints the time).
- Each proposal should name one of the five actions and cite three tags.
- The final state should be `TASK_STATE_COMPLETED`.

If a request is slow or fails, see Troubleshooting below before you spend your last check.

## Step 7: Submit to the grader

Give the grader your base URL from Step 5 and a Bearer token. Use your last check for Save only after Step 6 passes.

## Settings you can change (Render → Environment)

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | Claude model. Switch to `claude-sonnet-5-5` or `claude-haiku-5-5` if speed is a problem (cheaper, faster, less accurate). |
| `ANTHROPIC_EFFORT` | `medium` | How hard Claude thinks. `low` is faster. |
| `DECISION_BUDGET_SECONDS` | `38` | Model time allowed inside one request. Keep under 45. |
| `DAILY_MODEL_CALL_LIMIT` | `400` | Safety cap on Claude calls per day, so a public URL cannot run up a bill. |
| `A2A_ALLOWED_TOKENS` | unset | Comma-separated tokens allowed to call the agent. **Unset means any non-empty Bearer token works**, and every token is its own user. Set it before you go public if you do not want strangers using your API credit. |

## Troubleshooting

- **503 "Temporary model failure"**: Claude timed out, hit an API error, or the daily cap is reached. Nothing was saved, so retry with the same message ID. Check Render logs and the `ANTHROPIC_API_KEY`.
- **400 "Results must cover every proposal"**: a continuation must include one result for every proposal, in any order.
- **401**: missing or empty `Authorization: Bearer ...` header.
- **415**: `Content-Type` must be `application/a2a+json`.
- **400 version error**: send header `A2A-Version: 1.0`.
- **Only one Render instance**: keep it at one instance. The database is a single SQLite file on one disk, and duplicate-request waiting is kept in memory.

## Decisions I made (check these)

- **Reference tags keep their brackets.** `evidenceRefs` contains the tag exactly as it appears in the document, such as `[INV-A]`, including the brackets. If the grader expects the text without brackets, change `REF_PATTERN` in `app/decider.py`.
- **Receipts are a second artifact.** The proposal artifact is kept unchanged. The receipts go in a new artifact with one data part.
- **Model calls use chunks of six packages, running side by side.** This keeps each call well inside the 45-second limit. The grader only sees the results, not the chunking.
- **`returnImmediately` and `historyLength` are ignored.** Every send waits for the proposal step to finish, and the history is short, so both are safe to ignore.
- **Decision priority when several actions fit.** reject_duplicate, then open_exception, then hold_invoice, then request_approval, then settle_invoice. This is written in the system prompt in `app/decider.py`. Adjust it if your policy says otherwise.
- **Cover sheets, archive examples and training decoys are never cited.** The check is a text heuristic (it looks for those words in the paragraph that holds the reference).

## Not verified

- The Claude call was checked against a mocked HTTP transport and a fake decider. It has **not** been run against the live Claude API from this machine (no key was available here). Step 6 is the first real test.
- The grader itself was not run. Its exact request shapes are inferred from the assignment text.
