#!/usr/bin/env python3
"""News Briefing Agent

An LLM-assisted terminal agent that turns a natural-language request into a
validated news-fetching action, ranks current articles, and produces a cited
briefing. News is retrieved from public RSS feeds, so a separate news API key
is not required.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse

import feedparser
import requests
from dotenv import load_dotenv


LOG_FILE = Path("logs/agent_audit.jsonl")
REPORT_DIR = Path("reports")


@dataclass
class Article:
    title: str
    url: str
    source: str
    published: str
    summary: str


def audit(event: str, details: dict[str, Any]) -> None:
    """Write an audit event without storing API keys or private prompts."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "details": details,
    }
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_llm(_client: Any, prompt: str, *, json_mode: bool = False) -> str:
    """Call Gemini's REST endpoint with a bounded network timeout."""
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    if json_mode:
        payload["generationConfig"] = {"responseMimeType": "application/json"}
    for attempt in range(3):
        try:
            response = requests.post(
                endpoint,
                headers={"x-goog-api-key": api_key or "", "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:500]}")
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as error:
            audit("llm_error", {"attempt": attempt + 1, "error": str(error)[:300]})
            if attempt == 2:
                raise RuntimeError(f"The LLM request failed after three attempts: {error}") from error
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable")


def parse_request(client: Any, request: str) -> dict[str, Any]:
    """Have the LLM convert natural language into a small, safe action plan."""
    prompt = f"""You are the planning component of a news briefing agent.
Convert the user's request into JSON with exactly these fields:
{{
  "action": "fetch_news",
  "topic": "short search topic",
  "recency_hours": integer from 1 to 720,
  "trending": boolean,
  "important": boolean,
  "article_limit": integer from 3 to 15
}}

Interpret words such as latest/today/this week as recency. Interpret trending,
popular, or gaining attention as trending=true. Interpret important, major,
breaking, or significant as important=true. Do not create shell commands,
URLs, file paths, or any action other than fetch_news.

User request: {request}
"""
    raw = call_llm(client, prompt, json_mode=True)
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("The LLM returned an invalid action plan.") from error
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    """Reject anything outside the agent's explicitly allowed action schema."""
    if plan.get("action") != "fetch_news":
        raise ValueError("Safety check failed: unsupported action.")
    if not isinstance(plan.get("topic"), str) or not plan["topic"].strip():
        raise ValueError("Safety check failed: topic is missing.")
    if not 1 <= int(plan.get("recency_hours", 48)) <= 720:
        raise ValueError("Safety check failed: recency must be between 1 and 720 hours.")
    if not 3 <= int(plan.get("article_limit", 8)) <= 15:
        raise ValueError("Safety check failed: article limit must be between 3 and 15.")
    if not isinstance(plan.get("trending"), bool) or not isinstance(plan.get("important"), bool):
        raise ValueError("Safety check failed: ranking preferences must be boolean.")


def source_name(entry: Any) -> str:
    source = getattr(entry, "source", None)
    if isinstance(source, dict):
        return source.get("title", "Unknown source")
    link = getattr(entry, "link", "")
    return urlparse(link).netloc.replace("www.", "") or "Unknown source"


def parse_date(entry: Any) -> datetime:
    value = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if value:
        return datetime(*value[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - timedelta(days=365)


def fetch_news(plan: dict[str, Any]) -> list[Article]:
    """Execute the only external action allowed by this agent."""
    topic = plan["topic"].strip()
    query_terms = [topic]
    if plan["important"]:
        query_terms.append("(important OR major OR breaking)")
    query = quote_plus(" ".join(query_terms))
    feed_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    cutoff = datetime.now(timezone.utc) - timedelta(hours=int(plan["recency_hours"]))
    response = requests.get(feed_url, timeout=15, headers={"User-Agent": "NewsBriefingAgent/1.0"})
    response.raise_for_status()
    feed = feedparser.parse(response.content)
    articles: list[Article] = []
    seen_urls: set[str] = set()
    for entry in feed.entries:
        published = parse_date(entry)
        link = getattr(entry, "link", "")
        if not link or link in seen_urls or published < cutoff:
            continue
        seen_urls.add(link)
        articles.append(Article(
            title=re.sub(r"\s+", " ", getattr(entry, "title", "Untitled")).strip(),
            url=link,
            source=source_name(entry),
            published=published.isoformat(),
            summary=re.sub(r"<[^>]+>", "", getattr(entry, "summary", "")).strip()[:600],
        ))
    if not articles:
        raise RuntimeError("No recent articles were found. Try a broader topic or longer recency window.")
    return rank_articles(articles, plan)


def rank_articles(articles: list[Article], plan: dict[str, Any]) -> list[Article]:
    important_words = {"breaking", "major", "crisis", "election", "war", "launch", "report", "update"}
    source_counts: dict[str, int] = {}
    for article in articles:
        source_counts[article.source] = source_counts.get(article.source, 0) + 1

    def score(article: Article) -> float:
        age_hours = max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(article.published)).total_seconds() / 3600)
        freshness = max(0.0, 48 - age_hours) / 48
        title_words = set(re.findall(r"[a-z]+", article.title.lower()))
        importance = len(title_words & important_words) / 3
        diversity = 1 / source_counts[article.source]
        trending_score = diversity if plan["trending"] else 0
        return freshness + importance * (1 if plan["important"] else 0.25) + trending_score

    return sorted(articles, key=score, reverse=True)[: int(plan["article_limit"])]


def summarize(client: Any, plan: dict[str, Any], articles: list[Article]) -> str:
    article_text = "\n".join(
        f"[{index}] {article.title} | {article.source} | {article.published}\n{article.summary}\nURL: {article.url}"
        for index, article in enumerate(articles, 1)
    )
    prompt = f"""Create a concise, neutral news briefing about: {plan['topic']}.
The user requested a {plan['recency_hours']}-hour window.
Use only the articles below. Do not invent facts. Clearly distinguish reported
facts from uncertainty or disagreement. Include:
1. A two-sentence overview.
2. Three to five key developments with [article number] citations.
3. Why this matters.
4. A short 'What to watch next' section.
5. A Sources section listing each cited article as Markdown links.
Mention when the available articles do not establish a claim.

Articles:
{article_text}
"""
    return call_llm(client, prompt)


def save_report(plan: dict[str, Any], briefing: str, articles: list[Article]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe_topic = re.sub(r"[^a-z0-9]+", "-", plan["topic"].lower()).strip("-")[:45] or "news"
    path = REPORT_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{safe_topic}.md"
    text = f"# News Briefing: {plan['topic']}\n\nGenerated: {datetime.now().astimezone().isoformat()}\n\n{briefing}\n\n## Retrieved Articles\n\n"
    text += "\n".join(f"- [{a.title}]({a.url}) — {a.source}" for a in articles)
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    # Let the project's .env file override an older Windows environment
    # variable that may still contain a placeholder or revoked key.
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Summarize current news using natural-language preferences.")
    parser.add_argument("request", nargs="*", help='Example: "AI regulation, important news from the last week"')
    args = parser.parse_args()
    request = " ".join(args.request).strip() or input("What news would you like summarized? ")
    if not request:
        print("A topic is required.", file=sys.stderr)
        return 2
    if not os.getenv("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY in .env before running the agent.", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Limit each Gemini HTTP request to 60 seconds so a stalled connection
    # returns a readable error instead of leaving the terminal frozen.
    client = None
    try:
        audit("request_received", {"request": request[:500]})
        print("Planning the news search...")
        plan = parse_request(client, request)
        audit("action_validated", {"action": plan["action"], "topic": plan["topic"], "preferences": plan})
        print(f"Searching for {plan['topic']} (last {plan['recency_hours']} hours)...")
        articles = fetch_news(plan)
        audit("action_executed", {"action": "fetch_news", "article_count": len(articles)})
        print("Generating the briefing...")
        briefing = summarize(client, plan, articles)
        report_path = save_report(plan, briefing, articles)
        audit("report_created", {"path": str(report_path), "article_count": len(articles)})
        print("\n" + briefing)
        print(f"\nSaved report: {report_path}")
        return 0
    except Exception as error:
        audit("agent_error", {"error": str(error)[:500]})
        print(f"Agent error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
