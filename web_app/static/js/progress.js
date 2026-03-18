/**
 * Progress tracking via WebSocket for query execution.
 *
 * Connects to the WebSocket endpoint and receives real-time
 * progress updates during query execution.
 */

let progressSocket = null;
let pollInterval = null;
let currentQueryId = null;

/**
 * Initialize progress tracking for a query.
 *
 * @param {string} queryId - The UUID of the query to track.
 */
function initProgressTracking(queryId) {
    currentQueryId = queryId;

    // Try WebSocket first, fall back to polling
    if (window.WebSocket) {
        initWebSocket(queryId);
    } else {
        initPolling(queryId);
    }

    // Start query execution
    startQueryExecution(queryId);
}

/**
 * Initialize WebSocket connection for real-time updates.
 *
 * @param {string} queryId - The UUID of the query to track.
 */
function initWebSocket(queryId) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/query/${queryId}/`;

    try {
        progressSocket = new WebSocket(wsUrl);

        progressSocket.onopen = function(e) {
            console.log('WebSocket connected for query:', queryId);
        };

        progressSocket.onmessage = function(e) {
            const data = JSON.parse(e.data);
            handleProgressMessage(data);
        };

        progressSocket.onclose = function(e) {
            console.log('WebSocket closed:', e.code, e.reason);
            // Fall back to polling if WebSocket closes unexpectedly
            if (e.code !== 1000) {
                initPolling(queryId);
            }
        };

        progressSocket.onerror = function(e) {
            console.error('WebSocket error:', e);
            // Fall back to polling on error
            initPolling(queryId);
        };
    } catch (error) {
        console.error('WebSocket initialization error:', error);
        initPolling(queryId);
    }
}

/**
 * Initialize polling fallback for progress updates.
 *
 * @param {string} queryId - The UUID of the query to track.
 */
function initPolling(queryId) {
    if (pollInterval) {
        clearInterval(pollInterval);
    }

    pollInterval = setInterval(function() {
        fetchQueryStatus(queryId);
    }, 1000);
}

/**
 * Fetch query status via API (polling fallback).
 *
 * @param {string} queryId - The UUID of the query to check.
 */
async function fetchQueryStatus(queryId) {
    try {
        const response = await fetch(`/api/v1/query/${queryId}/status/`);
        const data = await response.json();

        handleProgressMessage({
            type: data.status === 'complete' ? 'complete' : 'progress',
            step: data.current_step,
            percent: data.progress,
            result: data.result,
        });

        // Stop polling if complete
        if (data.status === 'complete' || data.status === 'failed') {
            clearInterval(pollInterval);
            pollInterval = null;
        }
    } catch (error) {
        console.error('Error fetching query status:', error);
    }
}

/**
 * Handle progress message from WebSocket or polling.
 *
 * @param {Object} data - The progress data.
 */
function handleProgressMessage(data) {
    switch (data.type) {
        case 'connected':
            console.log('Progress tracking connected:', data.message);
            break;

        case 'progress':
            updateProgress(data.percent || 0, data.description || data.step || 'Processing...');

            // Update sub-queries display if available
            if (data.sub_queries && data.sub_queries.length > 0) {
                displaySubQueryProgress(data.sub_queries);
            }
            break;

        case 'complete':
            updateProgress(100, 'Complete');
            hideProgress();
            displayResults(data.result);
            break;

        case 'error':
            updateProgress(0, `Error: ${data.error}`);
            displayError(data.error, data.details);
            break;

        case 'pong':
            // Heartbeat response
            break;

        default:
            console.log('Unknown message type:', data.type, data);
    }
}

/**
 * Start query execution via API.
 *
 * @param {string} queryId - The UUID of the query.
 */
async function startQueryExecution(queryId) {
    // Get query text from session or page
    const queryText = window.queryText || '';

    updateProgress(5, 'Starting query execution...');

    try {
        const response = await apiFetch('/api/v1/query/', {
            method: 'POST',
            body: JSON.stringify({
                query: queryText,
                query_id: queryId,
                async: true,
            }),
        });

        const data = await response.json();

        if (!data.success && data.error) {
            handleProgressMessage({
                type: 'error',
                error: data.error,
            });
        } else if (!data.async) {
            // Synchronous response - display results immediately
            handleProgressMessage({
                type: 'complete',
                result: data,
            });
        }
        // For async responses, wait for WebSocket/polling updates
    } catch (error) {
        console.error('Error starting query:', error);
        handleProgressMessage({
            type: 'error',
            error: 'Failed to start query execution',
            details: error.message,
        });
    }
}

/**
 * Display sub-query progress (for smart routing).
 *
 * @param {Array} subQueries - Array of sub-query status objects.
 */
function displaySubQueryProgress(subQueries) {
    // This could update a sub-query list in the UI
    console.log('Sub-query progress:', subQueries);
}

/**
 * Display query results.
 *
 * @param {Object} result - The query result data.
 */
function displayResults(result) {
    // Hide loading state
    const loadingState = document.getElementById('loading-state');
    if (loadingState) {
        loadingState.style.display = 'none';
    }

    // Update query meta
    const queryMeta = document.getElementById('query-meta');
    if (queryMeta && result.execution_time) {
        queryMeta.textContent = `Completed in ${result.execution_time.toFixed(2)}s`;
    }

    // Update debug panel
    if (typeof updateDebugPanel === 'function') {
        updateDebugPanel(result);
    }

    // Display answer
    if (result.answer && typeof displayAnswer === 'function') {
        displayAnswer(result.answer, result.answer_type);
    }

    // Display chart if present
    if (result.figures && result.figures.length > 0 && typeof displayChart === 'function') {
        displayChart(result.figures[0]);
    } else if (result.metadata && result.metadata.has_figures && typeof displayChart === 'function') {
        // Try to extract chart from metadata
        if (result.metadata.figure) {
            displayChart({ figure: result.metadata.figure });
        }
    }

    // Display sources
    if (result.sources && result.sources.length > 0 && typeof displaySources === 'function') {
        displaySources(result.sources);
    }
}

/**
 * Display error state.
 *
 * @param {string} error - The error message.
 * @param {string} details - Optional error details.
 */
function displayError(error, details) {
    // Hide loading state
    const loadingState = document.getElementById('loading-state');
    if (loadingState) {
        loadingState.style.display = 'none';
    }

    // Show answer section with error
    const answerSection = document.getElementById('answer-section');
    const answerContent = document.getElementById('answer-content');

    if (answerSection && answerContent) {
        answerSection.style.display = 'block';
        answerContent.innerHTML = `
            <div style="color: var(--abbvie-red);">
                <h3>Error</h3>
                <p>${error}</p>
                ${details ? `<p class="text-muted">${details}</p>` : ''}
            </div>
        `;
    }
}

/**
 * Clean up WebSocket connection and polling.
 */
function cleanup() {
    if (progressSocket) {
        progressSocket.close();
        progressSocket = null;
    }
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// Clean up on page unload
window.addEventListener('beforeunload', cleanup);
