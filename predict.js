let faustModel = null;
let faustWordIndex = null;  // Wort->Index
let faustIdx2Word = null;   // Index->Wort
let seqLen = 20;

async function loadFaustModel() {
  // 1) Config laden
  const cfgRes = await fetch('config.json');
  const cfg = await cfgRes.json();
  seqLen = cfg.seqLen;
  console.log("✅ seqLen aus Config:", seqLen);

  // 2) Modell laden
  faustModel = await tf.loadLayersModel('web_model/model.json');
  console.log("✅ Modell geladen");

  // 3) Tokenizer laden
  const tokRes = await fetch('tokenizer.json');
  const tokenizerJson = await tokRes.json();
  const config = tokenizerJson['config'];

  // Direkt aus JSON
  faustWordIndex = JSON.parse(config['word_index']);    // Wort -> Index
  faustIdx2Word  = JSON.parse(config['index_word']);    // Index -> Wort

  console.log("Beispiel idx2Word[1]:", faustIdx2Word["1"]);  // sollte "<OOV>" zeigen
  console.log("Beispiel idx2Word[50]:", faustIdx2Word["50"]); // sollte "der" sein

  document.getElementById("status").classList.replace("alert-info", "alert-success");
  document.getElementById("status").innerText = "✅ Faust-Modell geladen!";
  document.getElementById('predictBtn').disabled = false;
  document.getElementById('continueBtn').disabled = false;
  document.getElementById('autoBtn').disabled = false;
  document.getElementById('resetBtn').disabled = false;
}

// Tokenize Seed → Indizes
function tokenizeSeed(seedText) {
  const tokens = seedText.toLowerCase().split(/\s+/).map(w => faustWordIndex[w] || 1); // 1 = <OOV>
  let context = tokens.slice(-seqLen);
  while (context.length < seqLen) context.unshift(0);
  return context;
}

// Top-K Vorhersagen holen
async function predictNextWordsBrowser(seedText, topK=5) {
  const context = tokenizeSeed(seedText);
  const inputTensor = tf.tensor2d([context], [1, seqLen], 'int32');
  const probs = faustModel.predict(inputTensor).dataSync();
  inputTensor.dispose();

  const sortedIdx = [...probs.keys()]
    .sort((a, b) => probs[b] - probs[a])
    .slice(0, topK);

  return sortedIdx.map(idx => ({
    word: faustIdx2Word[idx.toString()] || "<UNK>",
    prob: probs[idx]
  }));
}

// Greedy Top-1
async function predictBestWord(seedText) {
  const preds = await predictNextWordsBrowser(seedText, 1);
  return preds.length > 0 ? preds[0].word : "";
}

// Auto-Generierung
async function generateAutoText(seedText, steps=10) {
  let out = seedText;
  for (let i=0; i<steps; i++) {
    const nextWord = await predictBestWord(out);
    if (!nextWord) break;
    out += " " + nextWord;
  }
  return out;
}

// Start
loadFaustModel();
