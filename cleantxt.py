import re

with open("input.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = []

for line in lines:
    # Entfernt alles, was am Zeilenanfang wie "123<tab/space>" aussieht
    clean = re.sub(r'^\s*\d+\s+', '', line.strip())
    if clean:  # nur nicht-leere Zeilen behalten
        cleaned_lines.append(clean)

with open("cleaned.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(cleaned_lines))

print("✅ Nummern entfernt -> cleaned.txt")
