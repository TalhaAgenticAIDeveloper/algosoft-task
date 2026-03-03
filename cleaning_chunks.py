import json

# Input aur output file paths
input_file = "chunks.json"
output_file = "chunks_clean.json"

# Load the original JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Cleaned data list
cleaned_data = []

for chunk in data:
    cleaned_data.append({
        "chunk_id": chunk.get("chunkId"),  # original key c15, c16 etc
        "content": chunk.get("content")    # actual content text
    })

# Save cleaned data
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"✅ Cleaned {len(cleaned_data)} chunks. Saved to {output_file}")