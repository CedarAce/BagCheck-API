Build a baggage policy scraper pipeline in Python using Playwright and the Anthropic Claude API (claude-sonnet-4-20250514 with vision).
What it does: Reads a CSV of airlines (columns: id, name, policy_url) and for each airline, visits the baggage policy page, navigates through it intelligently, and extracts carry-on, personal item, and checked luggage dimensions and weight limits.
How it works:

Use Playwright (async, Chromium) to open each policy_url in a real browser
After the page loads, take a full-page screenshot
Send the screenshot to Claude claude-sonnet-4-20250514 vision with a prompt asking it to: (a) extract any baggage dimensions it can already see, and (b) identify any links, tabs, or buttons that likely lead to more specific baggage info (e.g. "Carry-on", "Checked bags", "Hand luggage") and return their visible label text
For each identified sub-link/tab, use Playwright to click it or navigate to it, take another screenshot, and send to Claude for extraction
Aggregate all extracted data per airline into a structured result: personal item dimensions (cm + inches), weight limit, carry-on dimensions, weight limit, checked bag dimensions, weight limit, and a confidence score ("high" / "medium" / "low") and notes field for anything ambiguous
After processing all airlines, write results to a CSV and optionally upsert into a Supabase table called airlines keyed on id

Error handling:

If a page fails to load or times out, mark that airline as status: failed and continue
If Claude can't find dimensions, set confidence: low and flag for manual review
Respect rate limits — add a short delay between airline requests

Config:

ANTHROPIC_API_KEY and SUPABASE_URL / SUPABASE_KEY from a .env file
A --dry-run flag that processes only the first 3 airlines for testing
A --airline flag to re-run a single airline by id

Output CSV columns: id, name, pi_dimensions_cm, pi_weight_kg, co_dimensions_cm, co_weight_kg, cb_dimensions_cm, cb_weight_kg, confidence, notes, last_scraped
