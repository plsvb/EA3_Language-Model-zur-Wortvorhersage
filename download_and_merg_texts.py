import os
import re
import requests
import zipfile
from io import BytesIO

# GitHub ZIP URL
ZIP_URL = "https://github.com/codealltag/CodEAlltag_pXL_GERMAN/archive/refs/heads/master.zip"

# Zielordner
BASE_DIR = "CodEAlltag_pXL_GERMAN-master"

def download_and_extract():
    print("📥 Lade Repo-ZIP von GitHub ...")
    r = requests.get(ZIP_URL)
    z = zipfile.ZipFile(BytesIO(r.content))
    z.extractall()
    print("✅ Repo entpackt!")

def clean_text(text):
    # lowercase + nur Buchstaben/Leerzeichen behalten
    text = text.lower()
    text = re.sub(r"[^a-zäöüß\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def merge_all_txt(base_dir):
    merged = []
    count = 0
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    raw = f.read()
                    merged.append(clean_text(raw))
                    count += 1
    return "\n".join(merged), count

# Schritt 1: Repo holen
if not os.path.exists(BASE_DIR):
    download_and_extract()

# Schritt 2: Alle .txt sammeln + bereinigen
print("🔍 Scanne Ordnerstruktur und sammle Mails ...")
final_text, total_files = merge_all_txt(BASE_DIR)

# Schritt 3: Zusammengeführte Trainingsdatei speichern
out_file = "training_data_alltag.txt"
with open(out_file, "w", encoding="utf-8") as out:
    out.write(final_text)

print(f"✅ Fertig! {total_files} Dateien zusammengeführt -> {out_file}")
