"""
AI-Powered Synthetic Dataset Generator for HS Code Classification
=================================================================
Uses Google Gemini to generate 252 high-quality, WCO-validated training samples
across 6 HS Code classes × 4 sample types:
  - standard    (20/class): Clear, unambiguous product descriptions
  - ambiguous   (10/class): Missing primary identifying keyword
  - overlapping  (6/class): Product at genuine boundary of 2 HS codes
  - edge_case    (6/class): Subject belongs to class A; modifier trap from class B

Output: synthetic_customs_data.csv (copied to both Method1_RAG and Method2_Traditional_ML)
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load API key from Method1_RAG/.env
env_path = Path(__file__).parent.parent / "Method1_RAG" / ".env"
load_dotenv(dotenv_path=env_path)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# HS Code Class Definitions (WCO-validated)
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = {
    "8517": {
        "category": "Telecommunications apparatus",
        "wco_description": "Telephone sets including smartphones; apparatus for transmission or reception of voice, images or data via wired or wireless network (routers, switches, modems, gateways)",
        "primary_keywords": ["smartphone", "telephone", "router", "switch", "modem", "gateway", "5G", "WiFi", "network hub"],
        "confusable_with": "8525",
        "overlap_rule": "WCO GRI Rule 1: Primary function is voice/data transmission (8517), not optical recording (8525), even if device has a high-quality camera.",
        "edge_trap_from": "8525",
        "edge_trap_keywords": ["camera", "megapixel", "optical zoom", "DSLR", "4K video", "lens"],
    },
    "8525": {
        "category": "Cameras and broadcast transmission apparatus",
        "wco_description": "Transmission apparatus for broadcasting; television cameras, digital cameras and video camera recorders, including action cameras, surveillance cameras, drone cameras",
        "primary_keywords": ["camera", "DSLR", "mirrorless", "megapixel", "optical", "video recorder", "broadcast", "lens"],
        "confusable_with": "8517",
        "overlap_rule": "WCO GRI Rule 1: Primary function is optical image capture/broadcast (8525), not voice/data transmission (8517), even if device has WiFi or streaming capability.",
        "edge_trap_from": "8517",
        "edge_trap_keywords": ["smartphone", "5G", "WiFi", "wireless network", "phone", "router"],
    },
    "6109": {
        "category": "T-shirts, singlets and vests — knitted or crocheted",
        "wco_description": "T-shirts, singlets and other vests, knitted or crocheted (HS Chapter 61). Fabric structure (knitted/crocheted) determines classification, not garment style.",
        "primary_keywords": ["T-shirt", "singlet", "vest", "knitted", "crocheted", "jersey", "crewneck"],
        "confusable_with": "6205",
        "overlap_rule": "WCO GRI Rule 1: Fabric construction (knitted jersey) takes precedence over garment style. A knitted polo shirt with collar and buttons is 6109, not 6205.",
        "edge_trap_from": "6205",
        "edge_trap_keywords": ["button-down", "Oxford", "woven", "dress shirt", "formal shirt", "collar"],
    },
    "6205": {
        "category": "Men's shirts — not knitted or crocheted (woven)",
        "wco_description": "Men's or boys' shirts, not knitted or crocheted (HS Chapter 62). Classification requires the fabric to be woven; any knitted construction reverts to Chapter 61.",
        "primary_keywords": ["shirt", "dress shirt", "button-down", "Oxford", "flannel", "linen shirt", "not knitted"],
        "confusable_with": "6109",
        "overlap_rule": "WCO GRI Rule 1: A garment described as a 'shirt' in woven fabric is 6205. However, if constructed from jersey or knitted fabric, it reverts to 6109 regardless of styling.",
        "edge_trap_from": "6109",
        "edge_trap_keywords": ["knitted", "jersey", "T-shirt", "crewneck", "crocheted", "singlet"],
    },
    "9403": {
        "category": "Other furniture and parts thereof",
        "wco_description": "Other furniture and parts thereof (HS Chapter 94). Includes tables, desks, shelves, wardrobes, cabinets, and other furnishings of any material. Primary function (seating, storage, work surface) determines classification over material.",
        "primary_keywords": ["table", "desk", "shelf", "wardrobe", "bookcase", "cabinet", "furniture", "dresser"],
        "confusable_with": "3926",
        "overlap_rule": "WCO GRI Rule 1: Primary function (furniture/seating/storage as part of household furnishing) takes precedence over material. A plastic outdoor chair is 9403, not 3926.",
        "edge_trap_from": "3926",
        "edge_trap_keywords": ["plastic", "polypropylene", "molded", "PVC", "acrylic", "injection-molded"],
    },
    "3926": {
        "category": "Other articles of plastics",
        "wco_description": "Other articles of plastics and articles of other materials of headings 39.01 to 39.14. Includes containers, cases, organizers, bins, and miscellaneous plastic articles not classified elsewhere.",
        "primary_keywords": ["plastic", "polypropylene", "molded", "PVC", "acrylic", "container", "bin", "carrying case"],
        "confusable_with": "9403",
        "overlap_rule": "WCO GRI Rule 3(b): When a product could be both a furniture item and a plastic article, if the product does not serve a dedicated furniture function (seating, work surface), plastic article (3926) takes precedence.",
        "edge_trap_from": "9403",
        "edge_trap_keywords": ["table", "desk", "chair", "shelf", "furniture", "wardrobe", "cabinet"],
    },
}

SAMPLE_TYPE_COUNTS = {
    "standard": 20,
    "ambiguous": 10,
    "overlapping": 6,
    "edge_case": 6,
}

# ─────────────────────────────────────────────────────────────────────────────
# Gemini Caller
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(prompt: str, retries: int = 3) -> list[dict]:
    """Call Gemini and parse JSON list response, with rate-limit-aware retry."""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()
            # Strip markdown fences if present
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.startswith("```"):
                raw = raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            return json.loads(raw.strip())
        except Exception as e:
            err = str(e)
            # Detect rate limit (429) and extract suggested retry_delay
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = 65  # Default: wait 65s to reset 1-minute window
                import re
                match = re.search(r'seconds: (\d+)', err)
                if match:
                    wait = int(match.group(1)) + 5
                print(f"    [Rate limit] Waiting {wait}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
            else:
                print(f"    [Attempt {attempt+1}/{retries}] Error: {e}")
                time.sleep(10)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Builders
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_standard(hs_code: str, cls: dict, count: int) -> str:
    return f"""
You are a customs data engineer generating training data for an HS Code classifier.

Generate exactly {count} UNIQUE product descriptions that would be classified as HS Code {hs_code}.

Official WCO Description: {cls['wco_description']}
Primary identifying keywords (at least 2 must appear in each description): {', '.join(cls['primary_keywords'])}

Rules:
- Each description must be clearly and unambiguously classifiable as {hs_code} by any algorithm.
- Vary product types, brands, materials, sizes, colors, and use cases to guarantee uniqueness.
- Descriptions should be realistic customs declaration style (10-25 words).
- Do NOT include descriptions that overlap with other HS codes.

Return ONLY a raw JSON array (no markdown fences) in this exact format:
[
  {{
    "description": "...",
    "hs_code": "{hs_code}",
    "category": "{cls['category']}",
    "sample_type": "standard",
    "confusable_with": "",
    "classification_note": "Clear standard sample: contains primary identifying keywords."
  }}
]
"""


def build_prompt_ambiguous(hs_code: str, cls: dict, count: int) -> str:
    return f"""
You are a customs data engineer generating challenging training data for an HS Code classifier.

Generate exactly {count} UNIQUE product descriptions that a HUMAN EXPERT would classify as HS Code {hs_code},
BUT intentionally omit ALL primary identifying keywords listed below.

Official WCO Description: {cls['wco_description']}
Primary keywords to OMIT entirely: {', '.join(cls['primary_keywords'])}

Rules:
- A customs expert reading the description should still correctly classify it as {hs_code}.
- A keyword-based algorithm (like TF-IDF) should struggle because no primary identifier is present.
- Use functional descriptions, material descriptions, or use-case descriptions instead.
- Descriptions should be 10-20 words. Each description must be unique.
- Do NOT use any of the primary keywords listed above. Not even partially.

Return ONLY a raw JSON array (no markdown fences):
[
  {{
    "description": "...",
    "hs_code": "{hs_code}",
    "category": "{cls['category']}",
    "sample_type": "ambiguous",
    "confusable_with": "",
    "classification_note": "Ambiguous: missing primary identifier '{cls['primary_keywords'][0]}'; inferrable from functional context."
  }}
]
"""


def build_prompt_overlapping(hs_code: str, cls: dict, count: int) -> str:
    confusable = cls["confusable_with"]
    confusable_cls = CLASSES[confusable]
    return f"""
You are a customs data engineer generating challenging training data for an HS Code classifier.

Generate exactly {count} UNIQUE product descriptions for products that genuinely straddle the boundary 
between HS Code {hs_code} and HS Code {confusable}.

Ground Truth: {hs_code} ({cls['category']})
Competing Code: {confusable} ({confusable_cls['category']})
WCO Classification Rule Applied: {cls['overlap_rule']}

Rules:
- Each description must have real characteristics from BOTH HS codes.
- A human expert applies WCO rules to determine the ground truth is {hs_code}.
- The descriptions should make even experienced classifiers pause and think.
- Descriptions should be 12-22 words. Each must be unique.

Return ONLY a raw JSON array (no markdown fences):
[
  {{
    "description": "...",
    "hs_code": "{hs_code}",
    "category": "{cls['category']}",
    "sample_type": "overlapping",
    "confusable_with": "{confusable}",
    "classification_note": "{cls['overlap_rule']}"
  }}
]
"""


def build_prompt_edge_case(hs_code: str, cls: dict, count: int) -> str:
    trap_from = cls["edge_trap_from"]
    trap_cls = CLASSES[trap_from]
    return f"""
You are a customs data engineer generating adversarial training data for an HS Code classifier.

Generate exactly {count} UNIQUE product descriptions where:
- The GRAMMATICAL SUBJECT (main product) belongs to HS Code {hs_code} ({cls['category']})
- The MODIFIER / CONTEXT WORDS contain strong keywords from HS Code {trap_from} ({trap_cls['category']})

Trap keywords to include from {trap_from}: {', '.join(cls['edge_trap_keywords'])}

Example structure: "[product of {hs_code}] specifically designed for / featuring / compatible with [trap keywords from {trap_from}]"

Goal: A keyword-based algorithm (TF-IDF) should misclassify this as {trap_from}.
A grammar-aware LLM should correctly identify the subject as {hs_code}.

Rules:
- The subject noun MUST be a product of {hs_code}.
- The trap keywords must appear naturally in the modifier clause, not as the subject.
- Descriptions should be 15-25 words. Each must be unique.
- Do NOT invent products — use realistic customs declaration language.

Return ONLY a raw JSON array (no markdown fences):
[
  {{
    "description": "...",
    "hs_code": "{hs_code}",
    "category": "{cls['category']}",
    "sample_type": "edge_case",
    "confusable_with": "{trap_from}",
    "classification_note": "Edge case: subject is {hs_code} product; modifier contains {trap_from} trap keywords ({', '.join(cls['edge_trap_keywords'][:3])})."
  }}
]
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_BUILDERS = {
    "standard": build_prompt_standard,
    "ambiguous": build_prompt_ambiguous,
    "overlapping": build_prompt_overlapping,
    "edge_case": build_prompt_edge_case,
}

def generate_dataset():
    all_records = []
    total_calls = len(CLASSES) * len(SAMPLE_TYPE_COUNTS)
    call_num = 0

    for hs_code, cls in CLASSES.items():
        for sample_type, count in SAMPLE_TYPE_COUNTS.items():
            call_num += 1
            print(f"[{call_num}/{total_calls}] Generating {count} '{sample_type}' samples for HS {hs_code} ({cls['category']})...")

            prompt = PROMPT_BUILDERS[sample_type](hs_code, cls, count)
            records = call_gemini(prompt)

            if records:
                all_records.extend(records)
                print(f"    OK — {len(records)} records received.")
            else:
                print(f"    WARNING: No records returned for {hs_code}/{sample_type}. Skipping.")

            # Polite delay between API calls: 7s = stay under 10 RPM (Free Tier)
            time.sleep(7)

    df = pd.DataFrame(all_records)

    # Ensure required columns exist
    for col in ["description", "hs_code", "category", "sample_type", "confusable_with", "classification_note"]:
        if col not in df.columns:
            df[col] = ""

    # Drop true duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["description"])
    after = len(df)
    if before != after:
        print(f"\nDropped {before - after} duplicate descriptions.")

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save outputs
    out_m2 = Path(__file__).parent / "synthetic_customs_data.csv"
    out_m1 = Path(__file__).parent.parent / "Method1_RAG" / "synthetic_customs_data.csv"

    df.to_csv(out_m2, index=False)
    df.to_csv(out_m1, index=False)

    print(f"\n{'='*60}")
    print(f"Dataset generated successfully!")
    print(f"  Total samples   : {len(df)}")
    print(f"  Saved to        : {out_m2}")
    print(f"  Synced to       : {out_m1}")
    print(f"\nDistribution by sample_type:")
    print(df.groupby(["hs_code", "sample_type"]).size().to_string())
    print(f"{'='*60}\n")
    return df


if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found in Method1_RAG/.env")
        print("Please add your API key and re-run.")
        exit(1)

    generate_dataset()
