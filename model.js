// model.js

// Tokenisierung vorbereiten
let tokens = [];
let word2Idx = {};
let idx2Word = [];
let vocabSize = 0;

// Trainingsdaten-Arrays
let seqLen = 10;
let Xdata = [];
let Ydata = [];
let numExamples = 0;

// TensorFlow.js Modell
let model = null;

// Tokenize + Vokabular aufbauen
function prepareData() {
  let text = trainingText.replace(/[.,;!?]/g, "").toLowerCase();
  tokens = text.split(/\s+/).filter(w => w.length > 0);

  let nextIndex = 0;
  tokens.forEach(w => {
    if (!(w in word2Idx)) {
      word2Idx[w] = nextIndex;
      idx2Word[nextIndex] = w;
      nextIndex++;
    }
  });
  vocabSize = idx2Word.length;

  // Sequenzen erzeugen
  Xdata = [];
  Ydata = [];
  for (let i = 0; i < tokens.length - seqLen; i++) {
    const seq = tokens.slice(i, i + seqLen).map(w => word2Idx[w]);
    const nextWord = tokens[i + seqLen];
    Xdata.push(seq);
    Ydata.push(word2Idx[nextWord]);
  }
  numExamples = Xdata.length;
}

// LSTM-Modell erstellen
function createModel() {
  model = tf.sequential();
  model.add(tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: 64,
    inputLength: seqLen
  }));
  model.add(tf.layers.lstm({ units: 100, returnSequences: true }));
  model.add(tf.layers.lstm({ units: 100 }));
  model.add(tf.layers.dense({ units: vocabSize, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'sparseCategoricalCrossentropy'
  });
}

// Modell trainieren
async function trainModel() {
  let xs = tf.tensor2d(Xdata, [numExamples, seqLen], 'int32');
  xs = xs.toFloat(); // <-- wichtig: cast zu float32
  const ys = tf.tensor1d(Ydata, 'int32'); // Labels bleiben int32

  await model.fit(xs, ys, {
    epochs: 30,
    batchSize: 32,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        document.getElementById('status').innerText =
          `Training... Epoche ${epoch + 1} / 30 | Loss: ${logs.loss.toFixed(4)}`;
      }
    }
  });

  xs.dispose();
  ys.dispose();
}


// Hilfsfunktion: Vorhersage für Top-N nächste Wörter
function predictNextWords(inputTokens, topN = 5) {
  // letzten seqLen Wörter extrahieren
  let context = inputTokens.slice(-seqLen);
  if (context.length < seqLen) {
    // mit Dummy auffüllen
    const pad = Array(seqLen - context.length).fill(idx2Word[0]);
    context = pad.concat(context);
  }

  const contextIndices = context.map(w => word2Idx[w] ?? 0);
  const inputTensor = tf.tensor2d([contextIndices], [1, seqLen], 'int32');

  const output = model.predict(inputTensor);
  const probs = output.dataSync();

  inputTensor.dispose();
  output.dispose();

  // Top-N ermitteln
  let indices = probs.map((v, i) => i);
  indices.sort((a, b) => probs[b] - probs[a]);
  const top = indices.slice(0, topN).map(idx => ({
    word: idx2Word[idx],
    prob: probs[idx]
  }));
  return top;
}

// Modellmetriken (Top-k Acc & Perplexity)
function evaluateModel() {
  const xs = tf.tensor2d(Xdata, [numExamples, seqLen], 'int32');
  const preds = model.predict(xs);
  const predArray = preds.arraySync();

  let top1 = 0, top5 = 0, top10 = 0, top20 = 0, top100 = 0;
  let totalLoss = 0;

  for (let i = 0; i < numExamples; i++) {
    const trueIdx = Ydata[i];
    const probs = predArray[i];
    const p = Math.max(probs[trueIdx], 1e-8);
    totalLoss += -Math.log(p);

    const sortedIdx = probs.map((v, idx) => idx).sort((a, b) => probs[b] - probs[a]);
    if (sortedIdx[0] === trueIdx) top1++;
    if (sortedIdx.slice(0, 5).includes(trueIdx)) top5++;
    if (sortedIdx.slice(0, 10).includes(trueIdx)) top10++;
    if (sortedIdx.slice(0, 20).includes(trueIdx)) top20++;
    if (sortedIdx.slice(0, 100).includes(trueIdx)) top100++;
  }

  const ppl = Math.exp(totalLoss / numExamples);
  xs.dispose(); preds.dispose();

  return {
    acc1: (top1 * 100 / numExamples).toFixed(1),
    acc5: (top5 * 100 / numExamples).toFixed(1),
    acc10: (top10 * 100 / numExamples).toFixed(1),
    acc20: (top20 * 100 / numExamples).toFixed(1),
    acc100: (top100 * 100 / numExamples).toFixed(1),
    perplexity: ppl.toFixed(2)
  };
}

// Initialisierung direkt aufrufen
prepareData();
createModel();
