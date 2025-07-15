let model;                // das geladene TensorFlow.js Modell
let wordIndex = {};       // Wort -> Index Mapping
let indexToWord = [];     // Index -> Wort Mapping
const sequenceLength = 5; // Anzahl Wörter für den Eingabekontext
let autoTimer = null;     // für Auto-Weiter

// Hilfsfunktion: Eingabetext bereinigen
function cleanInputText(str) {
  let txt = str.toLowerCase();
  txt = txt.replace(/[^a-zäöüß\s]+/g, "");
  return txt.trim();
}

// Eingabetext in Sequenz von Indizes umwandeln
function textToSequence(inputText) {
  const cleaned = cleanInputText(inputText);
  if (!cleaned) return Array(sequenceLength).fill(0);
  const words = cleaned.split(/\s+/);
  let seq = words.map(w => wordIndex[w] || 0);

  if (seq.length < sequenceLength) {
    const pad = Array(sequenceLength - seq.length).fill(0);
    seq = pad.concat(seq);
  } else if (seq.length > sequenceLength) {
    seq = seq.slice(seq.length - sequenceLength);
  }
  return seq;
}

// Nächste Wörter vorhersagen
function predictNextWords(inputText, topK = 5) {
  if (!model) return [];
  const seq = textToSequence(inputText);
  const inputTensor = tf.tensor2d([seq], [1, sequenceLength], 'int32');
  const prediction = model.predict(inputTensor);
  const probs = prediction.dataSync();

  let probArray = [];
  for (let i = 0; i < probs.length; i++) {
    probArray.push({ index: i, prob: probs[i] });
  }
  probArray.sort((a, b) => b.prob - a.prob);

  const top = probArray.slice(0, topK);
  return top.map(obj => ({
    word: indexToWord[obj.index] || "",
    probability: obj.prob
  }));
}

// Diagramm aktualisieren
function updateChart(predictions) {
  if (!predictions || predictions.length === 0) {
    Plotly.purge("chart");
    return;
  }
  const labels = predictions.map(p => p.word);
  const percents = predictions.map(p => (p.probability * 100).toFixed(2));
  const data = [{
    x: labels,
    y: percents,
    type: 'bar',
    text: percents.map(val => val + "%"),
    textposition: 'auto'
  }];
  const layout = {
    margin: { t: 30, b: 30 },
    yaxis: { title: 'Wahrscheinlichkeit (%)' },
    xaxis: { title: 'Vorhergesagte Wörter' },
    title: 'Top-Vorhersagen'
  };
  Plotly.newPlot('chart', data, layout, { staticPlot: true });
}

// Eingabefeld um ein Wort erweitern
function appendWordToInput(word) {
  if (!word) return;
  const textArea = document.getElementById('inputText');
  let currentText = textArea.value.trim();
  textArea.value = currentText ? (currentText + " " + word) : word;
}

// Initialisierung: Modell & Vokabular laden
async function initApp() {
  const statusEl = document.getElementById('status');
  statusEl.innerText = "Lade Vokabular...";
  // Vokabular laden
  const vocabResp = await fetch('model/vocab.json');
  const vocabData = await vocabResp.json();
  wordIndex = vocabData.wordIndex;
  indexToWord = vocabData.indexToWord;
  statusEl.innerText = "Lade Modell...";
  // Modell laden
  model = await tf.loadLayersModel('model/model.json');
  statusEl.innerText = "Modell & Vokabular geladen.";
  enableButtons();
}

// Buttons aktivieren
function enableButtons() {
  document.getElementById('predictBtn').disabled = false;
  document.getElementById('nextBtn').disabled = false;
  document.getElementById('autoBtn').disabled = false;
  document.getElementById('resetBtn').disabled = false;
}

// Event-Handler registrieren
function setupEventHandlers() {
  const predictBtn = document.getElementById('predictBtn');
  const nextBtn = document.getElementById('nextBtn');
  const autoBtn = document.getElementById('autoBtn');
  const stopBtn = document.getElementById('stopBtn');
  const resetBtn = document.getElementById('resetBtn');
  const statusEl = document.getElementById('status');

  predictBtn.addEventListener('click', () => {
    const inputText = document.getElementById('inputText').value;
    const preds = predictNextWords(inputText, 5);
    updateChart(preds);
  });

  nextBtn.addEventListener('click', () => {
    const inputText = document.getElementById('inputText').value;
    const preds = predictNextWords(inputText, 1);
    if (preds.length > 0) {
      appendWordToInput(preds[0].word);
    }
    updateChart([]);
  });

  autoBtn.addEventListener('click', () => {
    autoBtn.disabled = true;
    predictBtn.disabled = true;
    nextBtn.disabled = true;
    resetBtn.disabled = true;
    stopBtn.disabled = false;
    statusEl.innerText = "Auto-Modus läuft...";
    autoTimer = setInterval(() => {
      const currText = document.getElementById('inputText').value;
      const preds = predictNextWords(currText, 1);
      if (preds.length > 0) appendWordToInput(preds[0].word);
    }, 500);
  });

  stopBtn.addEventListener('click', () => {
    if (autoTimer) clearInterval(autoTimer);
    autoTimer = null;
    predictBtn.disabled = false;
    nextBtn.disabled = false;
    resetBtn.disabled = false;
    autoBtn.disabled = false;
    stopBtn.disabled = true;
    statusEl.innerText = "Auto-Modus gestoppt.";
  });

  resetBtn.addEventListener('click', () => {
    document.getElementById('inputText').value = "";
    updateChart([]);
  });
}

// Start, wenn DOM geladen ist
document.addEventListener('DOMContentLoaded', () => {
  // Buttons deaktivieren bis Modell fertig
  document.getElementById('predictBtn').disabled = true;
  document.getElementById('nextBtn').disabled = true;
  document.getElementById('autoBtn').disabled = true;
  document.getElementById('stopBtn').disabled = true;
  document.getElementById('resetBtn').disabled = true;

  setupEventHandlers();
  initApp(); // startet Laden von Vokab + Modell
});
