// Load JSON and populate dashboard
fetch("js/mockData.json")
    .then(response => response.json())
    .then(data => {
        // Summary
        document.getElementById("summary").innerHTML =
            `<p>Anomalies: ${data.summary.anomalies} | Suggestions: ${data.summary.suggestions}</p>`;

        // Table
        let table = "<tr><th>Column</th><th>Issue</th><th>Confidence</th><th>Fix</th></tr>";
        data.issues.forEach(issue => {
            table += `<tr>
                  <td>${issue.column}</td>
                  <td>${issue.issue}</td>
                  <td>${(issue.confidence * 100).toFixed(1)}%</td>
                  <td><button class="btn-primary">${issue.fix}</button></td>
                </tr>`;
        });
        document.getElementById("anomaliesTable").innerHTML = table;

        // Charts
        renderMissingChart(data.missing);
        renderTrendChart(data.trend);
    });
