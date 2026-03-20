/**
 * Chart rendering utilities using Plotly.
 *
 * Provides functions for rendering various chart types
 * with AbbVie branding colors.
 */

// AbbVie color palette for charts
const ABBVIE_CHART_COLORS = [
    '#8A2ECC',  // Purple
    '#0066F5',  // Cobalt
    '#338700',  // Green
    '#CF451C',  // Red
    '#A86BDE',  // Light Purple
    '#00A1FF',  // Light Cobalt
    '#45AB00',  // Light Green
    '#F7634F',  // Light Red
    '#A6B5E0',  // Medium Blue
    '#DBA63D',  // Light Copper
];

// Dark theme layout for Plotly
const DARK_LAYOUT = {
    paper_bgcolor: '#1E1E1E',
    plot_bgcolor: '#1E1E1E',
    font: {
        color: '#FFFFFF',
        family: 'Roboto, Arial, sans-serif',
    },
    xaxis: {
        gridcolor: '#333',
        zerolinecolor: '#333',
        linecolor: '#333',
    },
    yaxis: {
        gridcolor: '#333',
        zerolinecolor: '#333',
        linecolor: '#333',
    },
    margin: {
        l: 60,
        r: 30,
        t: 50,
        b: 60,
    },
};

/**
 * Render a Plotly chart in the specified container.
 *
 * @param {string} containerId - The ID of the container element.
 * @param {Object} plotlyJson - The Plotly figure data (can be JSON string or object).
 */
function renderChart(containerId, plotlyJson) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error('Chart container not found:', containerId);
        return;
    }

    // Parse JSON string if needed
    let figureData = plotlyJson;
    if (typeof plotlyJson === 'string') {
        try {
            figureData = JSON.parse(plotlyJson);
        } catch (e) {
            console.error('Error parsing chart JSON:', e);
            return;
        }
    }

    // Merge with dark theme layout
    const layout = {
        ...DARK_LAYOUT,
        ...figureData.layout,
    };

    // Apply AbbVie colors if not specified
    const data = figureData.data.map((trace, idx) => {
        if (!trace.marker || !trace.marker.color) {
            return {
                ...trace,
                marker: {
                    ...trace.marker,
                    color: ABBVIE_CHART_COLORS[idx % ABBVIE_CHART_COLORS.length],
                },
            };
        }
        return trace;
    });

    Plotly.newPlot(container, data, layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        displaylogo: false,
    });
}

/**
 * Create a bar chart from data.
 *
 * @param {string} containerId - The ID of the container element.
 * @param {Object} data - Object with keys as labels and values as counts.
 * @param {string} title - Chart title.
 */
function createBarChart(containerId, data, title) {
    const trace = {
        type: 'bar',
        x: Object.keys(data),
        y: Object.values(data),
        marker: {
            color: ABBVIE_CHART_COLORS[0],
        },
    };

    const layout = {
        ...DARK_LAYOUT,
        title: title || 'Results',
    };

    const container = document.getElementById(containerId);
    if (container) {
        Plotly.newPlot(container, [trace], layout, { responsive: true });
    }
}

/**
 * Create a pie chart from data.
 *
 * @param {string} containerId - The ID of the container element.
 * @param {Object} data - Object with keys as labels and values as counts.
 * @param {string} title - Chart title.
 */
function createPieChart(containerId, data, title) {
    const trace = {
        type: 'pie',
        labels: Object.keys(data),
        values: Object.values(data),
        marker: {
            colors: ABBVIE_CHART_COLORS.slice(0, Object.keys(data).length),
        },
        textinfo: 'label+percent',
        hoverinfo: 'label+value+percent',
    };

    const layout = {
        ...DARK_LAYOUT,
        title: title || 'Distribution',
        showlegend: true,
        legend: {
            font: { color: '#FFFFFF' },
        },
    };

    const container = document.getElementById(containerId);
    if (container) {
        Plotly.newPlot(container, [trace], layout, { responsive: true });
    }
}

/**
 * Create a line chart from data.
 *
 * @param {string} containerId - The ID of the container element.
 * @param {Array} xData - X-axis data.
 * @param {Array} yData - Y-axis data.
 * @param {string} title - Chart title.
 */
function createLineChart(containerId, xData, yData, title) {
    const trace = {
        type: 'scatter',
        mode: 'lines+markers',
        x: xData,
        y: yData,
        line: {
            color: ABBVIE_CHART_COLORS[0],
            width: 2,
        },
        marker: {
            color: ABBVIE_CHART_COLORS[0],
            size: 8,
        },
    };

    const layout = {
        ...DARK_LAYOUT,
        title: title || 'Trend',
    };

    const container = document.getElementById(containerId);
    if (container) {
        Plotly.newPlot(container, [trace], layout, { responsive: true });
    }
}

/**
 * Auto-detect chart type and create appropriate visualization.
 *
 * @param {string} containerId - The ID of the container element.
 * @param {Object|Array} data - The data to visualize.
 * @param {string} title - Chart title.
 */
function autoChart(containerId, data, title) {
    if (!data) return;

    // If it's an array of objects, create a grouped bar chart
    if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'object') {
        const keys = Object.keys(data[0]).filter(k => typeof data[0][k] === 'number');
        if (keys.length > 0) {
            const traces = keys.map((key, idx) => ({
                type: 'bar',
                name: key,
                x: data.map((d, i) => d.name || d.label || `Item ${i + 1}`),
                y: data.map(d => d[key]),
                marker: {
                    color: ABBVIE_CHART_COLORS[idx % ABBVIE_CHART_COLORS.length],
                },
            }));

            const layout = {
                ...DARK_LAYOUT,
                title: title || 'Comparison',
                barmode: 'group',
            };

            const container = document.getElementById(containerId);
            if (container) {
                Plotly.newPlot(container, traces, layout, { responsive: true });
            }
            return;
        }
    }

    // If it's an object with numeric values, decide between pie and bar
    if (typeof data === 'object' && !Array.isArray(data)) {
        const numKeys = Object.keys(data).length;
        if (numKeys <= 6) {
            createPieChart(containerId, data, title);
        } else {
            createBarChart(containerId, data, title);
        }
    }
}

/**
 * Check if data is suitable for chart visualization.
 *
 * @param {any} data - The data to check.
 * @returns {boolean} True if data can be charted.
 */
function isChartableData(data) {
    if (!data) return false;

    // Object with numeric values
    if (typeof data === 'object' && !Array.isArray(data)) {
        const values = Object.values(data);
        return values.length > 0 && values.every(v => typeof v === 'number');
    }

    // Array of objects with numeric properties
    if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'object') {
        const keys = Object.keys(data[0]);
        return keys.some(k => typeof data[0][k] === 'number');
    }

    return false;
}
