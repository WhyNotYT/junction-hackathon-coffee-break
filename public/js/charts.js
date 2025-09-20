// Chart.js bar chart
function renderMissingChart(missingData) {
    const ctx = document.getElementById("missingChart").getContext("2d");
    new Chart(ctx, {
        type: "bar",
        data: {
            labels: Object.keys(missingData),
            datasets: [{
                label: "Missing Values",
                data: Object.values(missingData),
                backgroundColor: "#00205B"
            }]
        }
    });
}

// D3.js line chart with anomalies
function renderTrendChart(trendData) {
    const svg = d3.select("#trendChart"),
        width = +svg.attr("width"),
        height = +svg.attr("height");

    const x = d3.scaleLinear().domain([0, trendData.length - 1]).range([40, width - 20]);
    const y = d3.scaleLinear().domain([0, d3.max(trendData)]).range([height - 30, 20]);

    const line = d3.line()
        .x((d, i) => x(i))
        .y(d => y(d));

    svg.append("path")
        .datum(trendData)
        .attr("fill", "none")
        .attr("stroke", "#00205B")
        .attr("stroke-width", 2)
        .attr("d", line);

    // Highlight anomaly (spike at index 3 for demo)
    svg.append("circle")
        .attr("cx", x(3))
        .attr("cy", y(trendData[3]))
        .attr("r", 6)
        .attr("fill", "red");
}
