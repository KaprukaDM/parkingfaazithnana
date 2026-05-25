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

load_dotenv()  # reads .env file

app = Flask(__name__)

# Load API key once at startup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("\n  ⚠️  OPENAI_API_KEY not found in .env file!")
    print("  Create a .env file with: OPENAI_API_KEY=sk-your-key-here\n")

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

        print(f"  ✅ Extracted {len(records)} records\n")
        return jsonify({"records": records})

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse AI response as JSON"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
