import json
import re

# ==============================
# STEP 1: LOAD BROKEN FILE SAFELY
# ==============================

with open("LO.json", "r", encoding="utf-8") as f:
    content = f.read().strip()

# Remove starting and ending quotes if present
if content.startswith('"'):
    content = content[1:]

if content.endswith('"'):
    content = content[:-1]

# Wrap with array brackets if missing
content = content.strip()

if not content.startswith('['):
    content = "[" + content

if not content.endswith(']'):
    content = content + "]"

# Now safely parse
raw_los = json.loads(content)

print(f"Loaded {len(raw_los)} raw LOs")

# ==============================
# STEP 2: CLEAN & STRUCTURE DATA
# ==============================

cleaned = []
seen_ids = set()

for item in raw_los:

    domain_full = item["Domain"]
    lo_text = item["Learning Outcome"].strip()

    # -------- Extract Domain --------
    domain_match = re.search(r"Domain\s+\d+:\s*(.*?)\.\s*Subdomain", domain_full)
    subdomain_match = re.search(r"Subdomain\s+\d+\.\d+:\s*(.*)", domain_full)

    domain = domain_match.group(1).strip() if domain_match else ""
    subdomain = subdomain_match.group(1).strip() if subdomain_match else ""

    # -------- Extract LO ID --------
    id_match = re.search(r"\d+\.\d+\.\d+\.\d+\.\d+", lo_text)
    if not id_match:
        continue

    lo_id = id_match.group()

    # -------- Extract Description --------
    description = lo_text.replace("Learning Outcome", "")
    description = description.replace(lo_id, "")
    description = description.replace(":", "")
    description = description.strip()

    # Remove accidental duplicated sentence (electric charge case)
    description = re.sub(r'(.+?)\1+', r'\1', description)

    # -------- Remove Duplicates --------
    if lo_id in seen_ids:
        continue

    seen_ids.add(lo_id)

    cleaned.append({
        "lo_id": lo_id,
        "domain": domain,
        "subdomain": subdomain,
        "description": description
    })

# ==============================
# STEP 3: SAVE CLEAN FILE
# ==============================

with open("clean_los.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2)

print(f"✅ Clean file generated with {len(cleaned)} LOs → clean_los.json")