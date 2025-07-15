let lmModel = null;
let wordIndex = null;   // Wort -> Index
let idx2Word = null;    // Index -> Wort
let seqLen = 20;        // wird aus config.json geladen

// --------------------------------------------
// Modell + Tokenizer laden
// --------------------------------------------
async function loadLMModel() {
  try {
    // 1) Config laden (enthält seqLen)
    const cfgRes = await fetch('config.json');
    const cfg = await cfgRes.json();
    seqLen = cfg.seqLen;
    console.log("✅ seqLen aus Config:", seqLen);

    // 2) Modell laden
    lmModel = await tf.loadLayersModel('web_model/model.json');
    console.log("✅ Modell geladen");

    // 3) Tokenizer laden
    const tokRes = await fetch('tokenizer.json');
    const tokenizerJson = await tokRes.json();
    const config = tokenizerJson['config'];

    wordIndex = JSON.parse(config['word_index']);    // Wort -> Index
    idx2Word  = JSON.parse(config['index_word']);    // Index -> Wort

    console.log("Beispiel idx2Word[1]:", idx2Word["1"]);
    console.log("Beispiel idx2Word[50]:", idx2Word["50"]);
    document.getElementById('seqLenVal').innerText = seqLen;


    // Buttons still im Hintergrund aktivieren
    document.getElementById('predictBtn').disabled = false;
    document.getElementById('continueBtn').disabled = false;
    document.getElementById('autoBtn').disabled = false;
    document.getElementById('resetBtn').disabled = false;

    // Event triggern für UI (falls benötigt)
    document.dispatchEvent(new Event("modelLoaded"));

  } catch (err) {
    console.error("❌ Fehler beim Laden des Modells:", err);
    alert("Fehler beim Laden des Modells. Bitte prüfe, ob model.json und alle Shard-Dateien korrekt sind.");
  }
}

// --------------------------------------------
// Tokenize Seed → Index-Array
// --------------------------------------------
function tokenizeSeed(seedText) {
  const tokens = seedText
    .toLowerCase()
    .replace(/[^a-zäöüß\s]/g, '')   // gleiche Cleaning-Logik wie beim Training
    .split(/\s+/)
    .map(w => wordIndex[w] || 1); // 1 = <OOV>

  // nur die letzten seqLen nehmen
  let context = tokens.slice(-seqLen);

  // falls zu kurz: vorne mit 0 auffüllen
  while (context.length < seqLen) context.unshift(0);
  return context;
}

// --------------------------------------------
// Top-K Vorhersagen holen (Browser) + <OOV>-Filter
// --------------------------------------------
async function predictNextWordsBrowser(seedText, topK = 5) {
  const context = tokenizeSeed(seedText);

  const inputTensor = tf.tensor2d([context], [1, seqLen], 'int32');
  const probs = lmModel.predict(inputTensor).dataSync();
  inputTensor.dispose();

  // Wahrscheinlichkeiten sortieren
  const sortedIdx = [...probs.keys()].sort((a, b) => probs[b] - probs[a]);

  // Mapping + <OOV> rausfiltern
  const predictions = [];
  for (let idx of sortedIdx) {
    const w = idx2Word[idx.toString()] || "<UNK>";
    if (w !== "<OOV>") {              // <OOV> nicht anzeigen
      predictions.push({ word: w, prob: probs[idx] });
    }
    if (predictions.length >= topK) break; // nur topK behalten
  }
  return predictions;
}

// --------------------------------------------
// Greedy Top-1 (bestes Wort)
// --------------------------------------------
async function predictBestWord(seedText) {
  const preds = await predictNextWordsBrowser(seedText, 1);
  return preds.length > 0 ? preds[0].word : "";
}

// --------------------------------------------
// Auto-Generierung (bis zu steps Wörter)
// --------------------------------------------
async function generateAutoText(seedText, steps = 10) {
  let out = seedText;
  for (let i = 0; i < steps; i++) {
    const nextWord = await predictBestWord(out);
    if (!nextWord) break;
    out += " " + nextWord;
  }
  return out;
}

// --------------------------------------------
// Start direkt beim Laden der Seite
// --------------------------------------------
loadLMModel();
