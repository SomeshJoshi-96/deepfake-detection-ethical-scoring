/**
 * Create a new ethical score chart
 * @param {string} canvasId - The ID of the canvas element
 * @param {number} score - The ethical score (0-100)
 * @returns {Chart} - The created Chart.js instance
 */
function createEthicalChart(canvasId, score) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Get appropriate color based on score
    const color = getEthicalScoreColor(score);
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [score, 100 - score],
                backgroundColor: [color, '#e0e0e0'],
                borderWidth: 0
            }]
        },
        options: {
            cutout: '75%',
            responsive: true,
            maintainAspectRatio: true,
            animation: {
                animateRotate: true,
                animateScale: true
            },
            plugins: {
                tooltip: {
                    enabled: false
                },
                legend: {
                    display: false
                }
            }
        }
    });
}

/**
 * Update an existing ethical score chart
 * @param {Chart} chart - The Chart.js instance to update
 * @param {number} score - The new ethical score (0-100)
 */
function updateEthicalChart(chart, score) {
    // Get appropriate color based on score
    const color = getEthicalScoreColor(score);
    
    // Update chart data
    chart.data.datasets[0].data[0] = score;
    chart.data.datasets[0].data[1] = 100 - score;
    chart.data.datasets[0].backgroundColor[0] = color;
    
    // Update chart
    chart.update();
}

/**
 * Get appropriate color based on ethical score value
 * @param {number} score - The ethical score (0-100)
 * @returns {string} - The color hex code
 */
function getEthicalScoreColor(score) {
    if (score < 30) {
        return '#34a853';  // Green for low concern
    } else if (score < 70) {
        return '#fbbc05';  // Yellow for moderate concern
    } else {
        return '#ea4335';  // Red for high concern
    }
}

/**
 * Get ethical impact text based on the score
 * @param {number} score - The ethical score (0-100)
 * @returns {string} - Descriptive text about the ethical impact
 */
function getEthicalImpactText(score) {
    if (score < 30) {
        return 'Low concern - Minor manipulation with limited potential harm.';
    } else if (score < 70) {
        return 'Moderate concern - Significant manipulation with moderate ethical impact.';
    } else {
        return 'High concern - Severe manipulation with significant potential for harm.';
    }
}
