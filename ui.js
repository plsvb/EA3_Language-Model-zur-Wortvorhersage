function updateChart(predictions) {
  const svg = d3.select("#chart");
  svg.selectAll("*").remove();

  const width = 400;
  const barHeight = 25;
  const margin = { top: 10, right: 50, bottom: 10, left: 100 };
  const height = predictions.length * (barHeight + 5);

  svg.attr("width", width + margin.left + margin.right)
     .attr("height", height + margin.top + margin.bottom);

  const x = d3.scaleLinear()
    .domain([0, d3.max(predictions, d => d.prob)])
    .range([0, width]);

  const g = svg.append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const bars = g.selectAll("g.bar")
    .data(predictions)
    .enter().append("g")
    .attr("class", "bar")
    .attr("transform", (d, i) => `translate(0, ${i * (barHeight + 5)})`);

  bars.append("rect")
    .attr("width", 0)
    .attr("height", barHeight)
    .attr("rx", 5).attr("ry", 5) // abgerundete Ecken
    .style("fill", (d, i) => d3.interpolateBlues(0.3 + i * 0.1)) // Farbverlauf
    .transition()
    .duration(600)
    .attr("width", d => x(d.prob));

  // Wortlabels links
  bars.append("text")
    .attr("x", -10)
    .attr("y", barHeight / 2)
    .attr("dy", ".35em")
    .attr("text-anchor", "end")
    .style("font-size", "14px")
    .style("fill", "#333")
    .text(d => d.word);

  // Prozentwerte rechts
  bars.append("text")
    .attr("x", d => x(d.prob) + 5)
    .attr("y", barHeight / 2)
    .attr("dy", ".35em")
    .style("font-size", "13px")
    .style("fill", "#444")
    .text(d => (d.prob * 100).toFixed(1) + "%");
}
