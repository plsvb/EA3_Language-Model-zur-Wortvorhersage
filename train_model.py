import os
# Workaround für Keras 3: Legacy-Modus erzwingen (kompatibler für TF.js)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import re
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Session clearen, damit keine _1/_2 Namenssuffixe entstehen
tf.keras.backend.clear_session()

# ---------------------------

# ---------------------------
with open("cleaned.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"📖 Original Textlänge: {len(raw_text)} Zeichen")

# ---------------------------
# 2) Textbereinigung
# ---------------------------
def clean_text(text):
    text = text.lower()
    # Nur deutsche Buchstaben + Leerzeichen
    text = re.sub(r"[^a-zäöüß\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

cleaned_text = clean_text(raw_text)
print(f"✅ Bereinigte Textlänge: {len(cleaned_text)} Zeichen")

# ---------------------------
# 3) Tokenizer + Sequenzen
# ---------------------------
max_vocab_size = 10000    # nur die 10k häufigsten Wörter
seqLen = 20               # Kontextlänge -> muss im Browser identisch sein!

tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts([cleaned_text])

total_words = min(max_vocab_size, len(tokenizer.word_index)) + 1
print(f"✅ Vokabulargröße: {total_words}")

# Sequenzen erzeugen
input_sequences = []
for line in cleaned_text.split("."):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(seqLen, len(token_list)):
        n_gram_seq = token_list[i - seqLen : i + 1]  # seqLen Wörter + Zielwort
        input_sequences.append(n_gram_seq)

print(f"✅ Trainingssequenzen: {len(input_sequences)}")

# Padding auf gleiche Länge
input_sequences = np.array(input_sequences)
X = input_sequences[:, :-1]
y = input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

print(f"✅ X-Shape: {X.shape}, y-Shape: {y.shape}")

# ---------------------------
# 4) Modell definieren (mit explizitem InputLayer & simplen Namen)
# ---------------------------
model = Sequential(name="model")  # KEIN Präfix wie "faust_lstm" für TF.js

# Explizite InputLayer -> TF.js bekommt batch_input_shape
model.add(InputLayer(batch_input_shape=(None, seqLen), name="input_layer"))

model.add(Embedding(total_words, 128, name="embedding"))
model.add(LSTM(256, return_sequences=True, name="lstm1"))
model.add(LSTM(256, name="lstm2"))
model.add(Dense(total_words, activation='softmax', name="output_dense"))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ---------------------------
# 5) Training
# ---------------------------
print("🚀 Training startet...")
model.fit(X, y, epochs=20, batch_size=128, verbose=1)
print("✅ Training abgeschlossen!")

# ---------------------------
# 6) Tokenizer + Config speichern
# ---------------------------
with open("tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer.to_json())

with open("config.json", "w") as f:
    json.dump({"seqLen": seqLen}, f)

print("✅ Tokenizer + Config gespeichert!")

# ---------------------------
# 7) Modell speichern (.keras + optional .h5)
# ---------------------------
# Modernes Keras-Format (empfohlen für TF.js)
model.save("faust_model.keras")
print("✅ Modell im .keras-Format gespeichert")

# Optional Backup im H5-Format
model.save("faust_model.h5")

# ---------------------------
# 8) Direkt ins TF.js-Webformat exportieren
# ---------------------------
print("🌐 Exportiere Modell ins Web-Format…")
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, "web_model")
print("✅ Web-Export abgeschlossen: Ordner 'web_model' enthält model.json + Shards!")

# Kurzer Check, ob batch_input_shape im JSON korrekt drin ist
with open("web_model/model.json", "r", encoding="utf-8") as f:
    model_json = json.load(f)
    topology = model_json.get("modelTopology", {})
    # nur prüfen, wenn es ein Keras-Layers-Modell ist
    if "config" in topology:
        first_layer = topology["config"]["layers"][0]
        print("🔍 Erster Layer nach Export:", first_layer["class_name"], first_layer["config"].get("batch_input_shape"))
    else:
        print("ℹ️ Exportiertes Modell ist evtl. Graph-Modell-Format, kein Layers-Config vorhanden – Überspringe Check.")
# ---------------------------
# 9) CLI-Funktionen (optional interaktiv testen)
# ---------------------------
def predict_next_word(seed_text):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=seqLen, padding='pre')
    predicted_idx = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]

    for word, idx in tokenizer.word_index.items():
        if idx == predicted_idx:
            return word
    return ""

def generate_text(seed_text, next_words=20):
    output = seed_text
    for _ in range(next_words):
        next_w = predict_next_word(output)
        if not next_w:
            break
        output += " " + next_w
    return output

def top_k_words(seed_text, k=5):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=seqLen, padding='pre')
    probs = model.predict(token_list, verbose=0)[0]
    sorted_idx = np.argsort(probs)[::-1][:k]
    idx2word = {idx: word for word, idx in tokenizer.word_index.items()}
    return [(idx2word.get(i, "<UNK>"), probs[i]) for i in sorted_idx]

# ---------------------------
# 10) Interaktive CLI (optional)
# ---------------------------
print("\nDu kannst Goethe interaktiv fragen!")
print("Tippe einen Starttext (oder 'exit' zum Beenden).")

while True:
    seed = input("\n> Dein Text: ").strip()
    if seed.lower() == "exit":
        break

    print("\nTop-5 wahrscheinlichste nächste Wörter:")
    for w, p in top_k_words(seed, k=5):
        print(f"  {w:<15} {p*100:.2f}%")

    suggestion = predict_next_word(seed)
    print(f"\n➡️  Vorschlag (Top-1): {suggestion}")

    print("\n📜 Auto-Fortsetzung (20 Wörter):")
    print(generate_text(seed, next_words=20))
