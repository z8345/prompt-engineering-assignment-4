# News Briefing Agent

A terminal-based AI agent that accepts a natural-language news request, retrieves recent articles, ranks them using recency, trending, and importance preferences, and asks an LLM to produce a cited summary.

## How it satisfies the project requirements

| Requirement | Implementation |
|---|---|
| LLM integration | Google Gemini's `google-genai` SDK is used for request planning and summarization. Retries handle temporary API failures. |
| Action interpreter/executor | The LLM creates a JSON `fetch_news` action. The program validates the schema, then executes the RSS retrieval and ranking operation. |
| User interface | A terminal interface accepts requests such as `"important AI news from the last week"`. |
| Error handling and safety | Actions are allow-listed, values are range-checked, API errors are caught, and no shell commands or destructive operations are permitted. |
| Auditability | Runs are appended to `logs/agent_audit.jsonl`, while reports are saved in `reports/`. |

## Setup

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
.\\.venv\\Scripts\\Activate.ps1
# Windows Command Prompt:
# .venv\\Scripts\\activate.bat

pip install -r requirements.txt
# macOS/Linux:
cp .env.example .env
# Windows PowerShell or Command Prompt:
# copy env.example .env
```

If PowerShell blocks the activation script, run this in the current PowerShell
window and then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\\.venv\\Scripts\\Activate.ps1
```

Put a Gemini API key in `.env`:

```text
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Create a Gemini key in [Google AI Studio](https://aistudio.google.com/apikey).
The program uses Google's official `google-genai` Python SDK. Keep the key in
`.env` and do not place it directly in the Python source code.

## Run examples

```bash
python news_agent.py "AI regulation, important news from the last 48 hours"
python news_agent.py "What is trending in cybersecurity this week?"
python news_agent.py "Major developments in renewable energy from the last month"
python news_agent.py
```

The agent uses Google News RSS for article discovery. It does not scrape article pages or bypass paywalls. Summaries are limited to the retrieved headlines and RSS descriptions, so the output should be treated as a briefing rather than a replacement for reading the original sources.

## Safety design

The LLM cannot directly run arbitrary code. Its response is parsed as JSON and must contain only the allow-listed `fetch_news` action. Topics are used only as URL-encoded search text. The executor performs a read-only HTTP request, and the only local writes are the generated Markdown report and audit log.

## Complexity

If `n` articles are returned, ranking takes `O(n log n)` because the articles are sorted. LLM and network time are external to the algorithm and depend on provider response time.
