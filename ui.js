let lastPredictions = [];
let autoRunning = false;
let autoTimeout = null;

// D3 Chart zeichnen
function drawChart(predictions) {
  const svg = d3.select("#chart");
  svg.selectAll("*").remove();

  if (!predictions || predictions.length === 0) return;

  const containerWidth = document.querySelector("#chart").clientWidth || 500;
  const barHeight = 28;
  const gap = 10;
  const labelOffset = 130;
  const height = predictions.length * (barHeight + gap);

  svg.attr("width", containerWidth).attr("height", height);

  const maxProb = d3.max(predictions, d => d.prob);
  const xScale = d3.scaleLinear()
    .domain([0, maxProb])
    .range([0, containerWidth - labelOffset - 50]);

  // Farbskala mit leichtem Verlauf
  const colorScale = d3.scaleLinear()
    .domain([0, predictions.length])
    .range(["#4da3ff", "#1a73e8"]);

  // Gruppe für jede Prediction
  const groups = svg.selectAll("g")
    .data(predictions)
    .enter()
    .append("g")
    .attr("transform", (d, i) => `translate(0, ${i * (barHeight + gap)})`)
    .style("cursor", "pointer")
    .on("click", (_, d) => addWordToInput(d.word))
    .on("mouseover", function () {
      d3.select(this).select("rect").attr("opacity", 0.8);
    })
    .on("mouseout", function () {
      d3.select(this).select("rect").attr("opacity", 1);
    });

  // Wortlabels links
  groups.append("text")
    .attr("x", 10)
    .attr("y", barHeight / 2)
    .attr("dy", ".35em")
    .style("font-size", "15px")
    .style("font-weight", "500")
    .style("fill", "#333")
    .text(d => d.word);

  // Balken mit Animation + abgerundeten Ecken
  groups.append("rect")
    .attr("x", labelOffset)
    .attr("y", 0)
    .attr("height", barHeight)
    .attr("rx", 6).attr("ry", 6)
    .style("fill", (d, i) => colorScale(i))
    .attr("width", 0) // Start bei 0 → dann animieren
    .transition()
    .duration(600)
    .attr("width", d => xScale(d.prob));

  // Prozentwerte am Balkenende
  groups.append("text")
    .attr("x", d => labelOffset + xScale(d.prob) + 8)
    .attr("y", barHeight / 2)
    .attr("dy", ".35em")
    .style("font-size", "14px")
    .style("fill", "#444")
    .text(d => (d.prob * 100).toFixed(1) + "%");
}


// Buttons-Logik
async function handlePrediction() {
  const text = document.getElementById("inputText").value.trim();
  if (!text) return;
  lastPredictions = await predictNextWordsBrowser(text, 5);
  drawChart(lastPredictions);
}

async function handleContinue() {
  const text = document.getElementById("inputText").value.trim();
  if (!text) return;
  const best = await predictBestWord(text);
  if (best) addWordToInput(best);
}

function addWordToInput(word) {
  const input = document.getElementById("inputText");
  let txt = input.value.trim();
  txt = txt.length > 0 ? txt + " " + word : word;
  input.value = txt;
  handlePrediction();
}

// Auto
async function startAuto() {
  if (!document.getElementById("inputText").value.trim()) {
    alert("Bitte einen Starttext eingeben!");
    return;
  }
  autoRunning = true;
  toggleButtonsDuringAuto(true);

  let count = 0;
  async function step() {
    if (!autoRunning || count >= 10) {
      stopAuto();
      return;
    }
    await handleContinue();
    count++;
    autoTimeout = setTimeout(step, 500);
  }
  step();
}

function stopAuto() {
  autoRunning = false;
  clearTimeout(autoTimeout);
  toggleButtonsDuringAuto(false);
}

function toggleButtonsDuringAuto(running) {
  document.getElementById("autoBtn").disabled = running;
  document.getElementById("stopBtn").disabled = !running;
  document.getElementById("predictBtn").disabled = running;
  document.getElementById("continueBtn").disabled = running;
}

function handleReset() {
  document.getElementById("inputText").value = "";
  lastPredictions = [];
  d3.select("#chart").selectAll("*").remove();
}

// Events
document.getElementById("predictBtn").addEventListener("click", handlePrediction);
document.getElementById("continueBtn").addEventListener("click", handleContinue);
document.getElementById("autoBtn").addEventListener("click", startAuto);
document.getElementById("stopBtn").addEventListener("click", stopAuto);
document.getElementById("resetBtn").addEventListener("click", handleReset);
