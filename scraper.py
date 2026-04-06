#!/usr/bin/env python3
"""
Baggage Policy Scraper Pipeline
--------------------------------
Reads a CSV of airlines (columns: id, name, policy_url), visits each baggage
policy page with a real Chromium browser, uses Claude vision to extract
carry-on, personal item, and checked-bag dimensions / weight limits, and
writes results to a CSV.

Usage:
  python scraper.py --input airlines.csv --output results.csv
  python scraper.py --dry-run
  python scraper.py --airline delta
"""

import argparse
import asyncio
import base64
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv
from playwright.async_api import BrowserContext, Page, async_playwright
from playwright_stealth import Stealth

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Model — override via MODEL env var, e.g. anthropic/claude-opus-4-5
MODEL = os.getenv("MODEL", "anthropic/claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

NAVIGATE_AND_EXTRACT_PROMPT = """\
You are analyzing a screenshot of an airline's baggage policy page.

Your task has TWO parts:

**PART 1 – Extract Baggage Dimensions & Limits**
Extract from what is currently visible on screen:
- Personal item (under-seat bag): dimensions (cm and/or inches) and weight limit
- Carry-on / cabin bag: dimensions (cm and/or inches) and weight limit
- Checked bag (first/standard bag): dimensions (cm and/or inches) and weight limit

**PART 2 – Identify Navigation Links / Tabs / Buttons**
Identify any visible links, tabs, accordion headings, or buttons that likely lead
to MORE specific baggage information. Examples: "Carry-on bags", "Checked bags",
"Hand luggage", "Baggage allowance", "Personal item", "Economy baggage", etc.
Return their EXACT visible label text as it appears on screen.

Return ONLY a JSON object with this exact schema (no markdown fences):
{
  "personal_item": {
    "dimensions_cm": "<string or null, e.g. '40 x 25 x 20 cm'>",
    "dimensions_in": "<string or null, e.g. '16 x 10 x 8 in'>",
    "weight_kg": <number or null>,
    "weight_lbs": <number or null>
  },
  "carry_on": {
    "dimensions_cm": "<string or null>",
    "dimensions_in": "<string or null>",
    "weight_kg": <number or null>,
    "weight_lbs": <number or null>
  },
  "checked_bag": {
    "dimensions_cm": "<string or null>",
    "dimensions_in": "<string or null>",
    "weight_kg": <number or null>,
    "weight_lbs": <number or null>
  },
  "navigation_labels": ["<exact visible label text>", ...],
  "confidence": "<high | medium | low>",
  "notes": "<any ambiguities, caveats, or manual-review flags>"
}

Confidence guide:
  "high"   – clear, explicit dimension tables are visible
  "medium" – some info found but incomplete or ambiguous
  "low"    – little or no baggage dimension info found

If a field is not visible on screen, use null. Be precise – include units as shown."""

EXTRACT_ONLY_PROMPT = """\
You are analyzing a screenshot of an airline's baggage policy sub-page.

Extract all baggage dimensions and weight limits visible on this page:
- Personal item (under-seat bag): dimensions (cm and/or inches) and weight limit
- Carry-on / cabin bag: dimensions (cm and/or inches) and weight limit
- Checked bag (first/standard bag): dimensions (cm and/or inches) and weight limit

Return ONLY a JSON object with this exact schema (no markdown fences):
{
  "personal_item": {
    "dimensions_cm": "<string or null>",
    "dimensions_in": "<string or null>",
    "weight_kg": <number or null>,
    "weight_lbs": <number or null>
  },
  "carry_on": {
    "dimensions_cm": "<string or null>",
    "dimensions_in": "<string or null>",
    "weight_kg": <number or null>,
    "weight_lbs": <number or null>
  },
  "checked_bag": {
    "dimensions_cm": "<string or null>",
    "dimensions_in": "<string or null>",
    "weight_kg": <number or null>,
    "weight_lbs": <number or null>
  },
  "confidence": "<high | medium | low>",
  "notes": "<any ambiguities or caveats>"
}

If a field is not visible on this sub-page, use null."""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AirlineResult:
    id: str
    name: str
    pi_dimensions_cm: Optional[str] = None
    pi_weight_kg: Optional[float] = None
    co_dimensions_cm: Optional[str] = None
    co_weight_kg: Optional[float] = None
    cb_dimensions_cm: Optional[str] = None
    cb_weight_kg: Optional[float] = None
    confidence: str = "low"
    notes: str = ""
    last_scraped: Optional[str] = None
    status: str = "ok"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def merge_extraction(result: AirlineResult, extraction: dict) -> None:
    """Merge a Claude extraction dict into AirlineResult, filling missing fields."""

    def _fill(current, candidate):
        """Use candidate only when current is unset."""
        return candidate if (current is None and candidate is not None) else current

    pi = extraction.get("personal_item") or {}
    result.pi_dimensions_cm = _fill(result.pi_dimensions_cm, pi.get("dimensions_cm"))
    result.pi_weight_kg = _fill(result.pi_weight_kg, pi.get("weight_kg"))

    co = extraction.get("carry_on") or {}
    result.co_dimensions_cm = _fill(result.co_dimensions_cm, co.get("dimensions_cm"))
    result.co_weight_kg = _fill(result.co_weight_kg, co.get("weight_kg"))

    cb = extraction.get("checked_bag") or {}
    result.cb_dimensions_cm = _fill(result.cb_dimensions_cm, cb.get("dimensions_cm"))
    result.cb_weight_kg = _fill(result.cb_weight_kg, cb.get("weight_kg"))

    # Upgrade confidence only
    new_conf = extraction.get("confidence", "low")
    if _CONFIDENCE_RANK.get(new_conf, 0) > _CONFIDENCE_RANK.get(result.confidence, 0):
        result.confidence = new_conf

    # Append notes (deduplicate)
    new_notes = (extraction.get("notes") or "").strip()
    if new_notes and new_notes not in result.notes:
        result.notes = f"{result.notes}; {new_notes}".lstrip("; ")


async def take_screenshot(page: Page, max_px: int = 7000) -> str:
    """Full-page screenshot → base64 PNG string, scaled down if either dimension exceeds max_px."""
    data = await page.screenshot(full_page=True)

    # Check dimensions and scale down if needed to stay within Claude's 8000px limit
    try:
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if w > max_px or h > max_px:
            scale = max_px / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            data = buf.getvalue()
            logger.debug(f"Screenshot resized {w}x{h} → {new_w}x{new_h}")
    except ImportError:
        # Pillow not installed — fall back to viewport-only screenshot
        logger.warning("Pillow not installed; falling back to viewport screenshot to avoid size limit")
        data = await page.screenshot(full_page=False)

    return base64.standard_b64encode(data).decode("utf-8")


def call_claude_vision(
    client: OpenAI, image_b64: str, prompt: str
) -> dict:
    """Send a screenshot to the model via OpenRouter and return parsed JSON dict."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    text = (response.choices[0].message.content or "").strip()

    # Strip markdown code fences if the model wraps the JSON anyway
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (```json or ```) and last line (```)
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Claude returned non-JSON; wrapping as low-confidence result")
        return {
            "confidence": "low",
            "notes": f"JSON parse error – raw output: {text[:300]}",
        }


async def try_click_label(page: Page, label: str) -> bool:
    """
    Try to click a visible element matching *label*.
    Returns True if an element was found and clicked.
    """
    # Playwright locator strategies, ordered by specificity
    strategies = [
        page.get_by_role("tab", name=label, exact=True),
        page.get_by_role("button", name=label, exact=True),
        page.get_by_role("link", name=label, exact=True),
        page.get_by_text(label, exact=True),
        # Partial-text fallback
        page.get_by_role("tab", name=label),
        page.get_by_role("button", name=label),
        page.get_by_role("link", name=label),
        page.get_by_text(label),
    ]

    for locator in strategies:
        try:
            if await locator.count() > 0:
                await locator.first.click(timeout=5000)
                # Wait for network to settle after click
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=6000)
                except Exception:
                    pass
                await asyncio.sleep(1.5)
                return True
        except Exception:
            continue

    return False


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

async def scrape_airline(
    airline_id: str,
    airline_name: str,
    policy_url: str,
    client: OpenAI,
    context: BrowserContext,
    page_timeout: int = 60_000,
    max_subnav: int = 5,
) -> AirlineResult:
    """
    Scrape one airline's baggage policy page and return an AirlineResult.
    Never raises – errors are captured in result.status / result.notes.
    """
    result = AirlineResult(id=airline_id, name=airline_name)
    page = await context.new_page()

    try:
        logger.info(f"[{airline_name}] → {policy_url}")
        await Stealth().apply_stealth_async(page)
        try:
            await page.goto(policy_url, wait_until="domcontentloaded", timeout=page_timeout)
        except Exception:
            # Some sites throw on domcontentloaded but the page still loads — try load
            await page.goto(policy_url, wait_until="load", timeout=page_timeout)
        # Extra settle time for JS-heavy SPAs
        await asyncio.sleep(3)

        # --- Step 1: initial screenshot + extract + nav discovery ---
        screenshot_b64 = await take_screenshot(page)
        extraction = call_claude_vision(client, screenshot_b64, NAVIGATE_AND_EXTRACT_PROMPT)
        merge_extraction(result, extraction)

        nav_labels: list[str] = [
            str(lbl).strip()
            for lbl in (extraction.get("navigation_labels") or [])
            if lbl and str(lbl).strip()
        ]
        logger.info(
            f"[{airline_name}] Initial confidence={result.confidence}  "
            f"nav_labels={nav_labels}"
        )

        # --- Step 2: follow each discovered navigation label ---
        visited: set[str] = set()
        for label in nav_labels[:max_subnav]:
            if label in visited:
                continue
            visited.add(label)

            logger.info(f"[{airline_name}]   clicking '{label}'")
            clicked = await try_click_label(page, label)
            if not clicked:
                logger.debug(f"[{airline_name}]   could not locate '{label}'")
                continue

            await asyncio.sleep(1.5)
            sub_b64 = await take_screenshot(page)
            sub_extraction = call_claude_vision(client, sub_b64, EXTRACT_ONLY_PROMPT)
            merge_extraction(result, sub_extraction)
            logger.info(
                f"[{airline_name}]   sub-page '{label}' → "
                f"confidence={sub_extraction.get('confidence')}"
            )

        result.last_scraped = datetime.now(timezone.utc).isoformat()

    except Exception as exc:
        logger.error(f"[{airline_name}] FAILED – {exc}")
        result.status = "failed"
        result.confidence = "low"
        result.notes = f"Scrape error: {str(exc)[:250]}"
        result.last_scraped = datetime.now(timezone.utc).isoformat()

    finally:
        await page.close()

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_FIELDS = [
    "id", "name",
    "pi_dimensions_cm", "pi_weight_kg",
    "co_dimensions_cm", "co_weight_kg",
    "cb_dimensions_cm", "cb_weight_kg",
    "confidence", "notes", "last_scraped",
]


def write_csv(results: list[AirlineResult], output_path: str) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({field: getattr(r, field, "") for field in OUTPUT_FIELDS})
    logger.info(f"Results written → {output_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Read input CSV
    with open(args.input, newline="", encoding="utf-8") as fh:
        airlines = list(csv.DictReader(fh))

    if not airlines:
        logger.error(f"No rows found in {args.input}")
        sys.exit(1)

    # Apply --airline filter
    if args.airline:
        airlines = [a for a in airlines if a.get("id", "").strip() == args.airline]
        if not airlines:
            logger.error(f"No airline with id='{args.airline}' found in {args.input}")
            sys.exit(1)

    # Apply --dry-run cap
    if args.dry_run:
        airlines = airlines[:3]
        logger.info("--dry-run: capped at first 3 airlines")

    logger.info(f"Processing {len(airlines)} airline(s) with model={MODEL}")

    results: list[AirlineResult] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=not args.headed,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            },
        )

        for idx, row in enumerate(airlines):
            airline_id = row.get("id", "").strip()
            airline_name = row.get("name", "").strip()
            policy_url = row.get("policy_url", "").strip()

            if not policy_url:
                logger.warning(f"Skipping '{airline_name}' – no policy_url")
                continue

            result = await scrape_airline(
                airline_id=airline_id,
                airline_name=airline_name,
                policy_url=policy_url,
                client=client,
                context=context,
                page_timeout=60_000,
            )
            results.append(result)

            # Rate limiting between airlines
            if idx < len(airlines) - 1:
                logger.info(f"Waiting {args.delay}s before next airline…")
                await asyncio.sleep(args.delay)

        await context.close()
        await browser.close()

    write_csv(results, args.output)

    # Summary
    n_ok = sum(1 for r in results if r.status == "ok")
    n_fail = sum(1 for r in results if r.status == "failed")
    conf_counts = {c: sum(1 for r in results if r.confidence == c) for c in ("high", "medium", "low")}
    logger.info(
        f"Done – {n_ok} ok / {n_fail} failed | "
        f"confidence: {conf_counts['high']} high, "
        f"{conf_counts['medium']} medium, "
        f"{conf_counts['low']} low"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape airline baggage policies with Playwright + Claude vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        default="airlines.csv",
        help="Input CSV file with columns: id, name, policy_url (default: airlines.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results.csv",
        help="Output CSV file path (default: results.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 3 airlines for testing",
    )
    parser.add_argument(
        "--airline",
        metavar="ID",
        default=None,
        help="Re-process a single airline by its id column value",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        metavar="SECONDS",
        help="Seconds to wait between airline requests (default: 3.0)",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show the browser window (useful for debugging blocked pages)",
    )

    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
