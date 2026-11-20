"""
Parking Data Extractor — Python Backend
Run: pip install flask openai python-dotenv
Then: python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
from openai import OpenAI
from dotenv import load_dotenv
import json
import os
import threading
from datetime import datetime, timezone

load_dotenv()  # reads .env file

app = Flask(__name__)

# Load API key once at startup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("\n  ⚠️  OPENAI_API_KEY not found in .env file!")
    print("  Create a .env file with: OPENAI_API_KEY=sk-your-key-here\n")

# -- Activity / cost tracking ------------------------------------------------
MODEL = "gpt-4o"
# USD per 1M tokens (gpt-4o list price) - update if OpenAI pricing changes
PRICE_INPUT_PER_M = 2.50
PRICE_OUTPUT_PER_M = 10.00
ACTIVITY_FILE = os.getenv("ACTIVITY_FILE", "activity_log.json")
_log_lock = threading.Lock()


def _load_activity():
    try:
        with open(ACTIVITY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


ACTIVITY = _load_activity()


def log_activity(entry):
    with _log_lock:
        ACTIVITY.append(entry)
        try:
            with open(ACTIVITY_FILE, "w", encoding="utf-8") as f:
                json.dump(ACTIVITY, f)
        except Exception as e:
            print(f"  Could not save activity log: {e}")


def calc_cost(prompt_tokens, completion_tokens):
    return (prompt_tokens * PRICE_INPUT_PER_M + completion_tokens * PRICE_OUTPUT_PER_M) / 1_000_000


FIELDS = [
    "reference","airport","terminalIn","terminalOut","customerName","mobileNo",
    "departureDate","departureTime","departureFlight","returnDate","returnTime",
    "returnFlight","registration","make","model","colour","days","price","status","email","product"
]

SYSTEM_PROMPT = f"""You are a strict, highly accurate data extraction engine for UK parking booking records.
Your priority is 100% fidelity to the source text. Do not invent information.
Return a JSON object with a single key "records" containing an array of objects.
Each object must have these keys: {", ".join(FIELDS)}.

Rules:
- Extract verbatim. Do NOT fix spelling unless it's a clear OCR error.
- If a field is missing, use an empty string "".
- Dates must be DD/MM/YYYY. Times must be HH:MM.
- Normalize terminals to "Terminal X" (e.g. T5 → Terminal 5).
- product must be "M&G" (Meet & Greet/Valet), "P&R" (Park & Ride/Shuttle), "EV" (Electric), or "".
- price must be numeric only, no currency symbols.
- Extract EVERY distinct booking found.
Return ONLY valid JSON like {{"records": [...]}}. No markdown, no backticks, no explanation."""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "api_key_loaded": bool(OPENAI_API_KEY)})


@app.route("/extract", methods=["POST"])
def extract():
    data = request.json or {}
    input_text = data.get("input_text", "").strip()
    image_b64 = data.get("image_base64")
    mime_type = data.get("mime_type")

    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY not set in .env file"}), 400
    if not input_text and not image_b64:
        return jsonify({"error": "Provide text or an image"}), 400

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Build user message content
        user_content = []

        if image_b64 and mime_type:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}
            })

        user_content.append({
            "type": "text",
            "text": f"Extract parking booking records from this input:\n\n{input_text or '(see attached image)'}"
        })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content or "{}"
        print(f"\n  📦 Raw OpenAI response:\n{raw[:500]}\n")  # debug log
        parsed = json.loads(raw)

        # Handle any response shape — object with any key, or direct array
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            # Try known keys first, then grab the first list value found
            records = (
                parsed.get("records")
                or parsed.get("bookings")
                or parsed.get("data")
                or next((v for v in parsed.values() if isinstance(v, list)), [])
            )
        else:
            records = []

        usage = response.usage
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0
        cost = calc_cost(pt, ct)
        log_activity({
            "time": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "has_text": bool(input_text),
            "has_image": bool(image_b64),
            "records": len(records),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cost_usd": cost,
            "status": "ok",
        })

        print(f"  ✅ Extracted {len(records)} records (${cost:.4f})\n")
        return jsonify({"records": records, "usage": {"prompt_tokens": pt, "completion_tokens": ct, "cost_usd": cost}})

    except json.JSONDecodeError:
        _log_failure(input_text, image_b64, "Failed to parse AI response as JSON")
        return jsonify({"error": "Failed to parse AI response as JSON"}), 500
    except Exception as e:
        _log_failure(input_text, image_b64, str(e))
        return jsonify({"error": str(e)}), 500


def _log_failure(input_text, image_b64, msg):
    log_activity({
        "time": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "has_text": bool(input_text),
        "has_image": bool(image_b64),
        "records": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "status": "error",
        "error": msg[:200],
    })


@app.route("/activity")
def activity():
    with _log_lock:
        entries = list(ACTIVITY)
    ok = [e for e in entries if e.get("status") == "ok"]
    return jsonify({
        "model": MODEL,
        "pricing": {"input_per_m": PRICE_INPUT_PER_M, "output_per_m": PRICE_OUTPUT_PER_M},
        "totals": {
            "sessions": len(entries),
            "records": sum(e.get("records", 0) for e in ok),
            "prompt_tokens": sum(e.get("prompt_tokens", 0) for e in ok),
            "completion_tokens": sum(e.get("completion_tokens", 0) for e in ok),
            "cost_usd": sum(e.get("cost_usd", 0) for e in ok),
        },
        "entries": entries[::-1],
    })


if __name__ == "__main__":
    print("\n  🅿️  Parking Data Extractor")
    print("  ─────────────────────────────────")
    print(f"  API Key: {'✅ Loaded' if OPENAI_API_KEY else '❌ Missing — check .env'}")
    print("  ─────────────────────────────────")
    print("  Open this URL in your browser:")
    print("  → http://localhost:5000")
    print("  ─────────────────────────────────")
    print("  Test: http://localhost:5000/ping")
    print()
    app.run(debug=True, port=5000)
