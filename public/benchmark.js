const BACKEND_URL = 'http://localhost:5500';
const STATUS_POLL_INTERVAL = 1000;

// DOM Elements
const configPanel = document.querySelector('.config-panel');
const statusPanel = document.getElementById('statusPanel');
const resultsPanel = document.getElementById('resultsPanel');
const startBenchmarkBtn = document.getElementById('startBenchmarkBtn');
const versionSelect = document.getElementById('versionSelect');
const problemCount = document.getElementById('problemCount');
const problemCountSlider = document.getElementById('problemCountSlider');
const difficultySelect = document.getElementById('difficultySelect');
const currentProblem = document.getElementById('currentProblem');
const progressText = document.getElementById('progressText');
const progressFill = document.getElementById('progressFill');
const timeElapsed = document.getElementById('timeElapsed');
const backendStatus = document.getElementById('backendStatus');
const overallMetrics = document.getElementById('overallMetrics');
const difficultyMetrics = document.getElementById('difficultyMetrics');
const platformMetrics = document.getElementById('platformMetrics');
const errorBreakdown = document.getElementById('errorBreakdown');
const exportResultsBtn = document.getElementById('exportResultsBtn');
const loadLatestResultsBtn = document.getElementById('loadLatestResultsBtn');
const viewResultsHistoryBtn = document.getElementById('viewResultsHistoryBtn');
const historyModal = document.getElementById('historyModal');
const closeHistoryBtn = document.getElementById('closeHistoryBtn');
const historyList = document.getElementById('historyList');

let benchmarkRunning = false;
let benchmarkStartTime = null;
let statusCheckInterval = null;
let currentResults = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  checkBackendStatus();
  setupEventListeners();
  linkSliderAndInput();
});

function setupEventListeners() {
  startBenchmarkBtn.addEventListener('click', startBenchmark);
  exportResultsBtn.addEventListener('click', exportResults);
  loadLatestResultsBtn.addEventListener('click', loadLatestResults);
  viewResultsHistoryBtn.addEventListener('click', showResultsHistory);
  closeHistoryBtn.addEventListener('click', closeResultsHistory);
  historyModal.addEventListener('click', (e) => {
    if (e.target === historyModal) closeResultsHistory();
  });
}

function linkSliderAndInput() {
  problemCount.addEventListener('change', (e) => {
    problemCountSlider.value = e.target.value;
  });
  problemCountSlider.addEventListener('input', (e) => {
    problemCount.value = e.target.value;
  });
}

async function checkBackendStatus() {
  try {
    const response = await fetch(`${BACKEND_URL}/benchmark/status`);
    if (response.ok) {
      backendStatus.textContent = 'Backend: connected';
      backendStatus.classList.add('connected');
    } else {
      backendStatus.textContent = 'Backend: error';
      backendStatus.classList.add('error');
    }
  } catch (error) {
    backendStatus.textContent = 'Backend: offline';
    backendStatus.classList.add('error');
  }
}

async function startBenchmark() {
  if (benchmarkRunning) {
    showAlert('Benchmark is already running', 'warning');
    return;
  }

  const n = parseInt(problemCount.value) || 10;
  const version = versionSelect.value;
  const difficulty = difficultySelect.value || null;

  if (n < 1 || n > 100) {
    showAlert('Problem count must be between 1 and 100', 'error');
    return;
  }

  try {
    startBenchmarkBtn.disabled = true;
    startBenchmarkBtn.textContent = 'Starting...';

    const response = await fetch(`${BACKEND_URL}/benchmark/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        version,
        n,
        difficulty,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to start benchmark');
    }

    benchmarkRunning = true;
    benchmarkStartTime = Date.now();
    configPanel.style.display = 'none';
    statusPanel.style.display = 'block';
    resultsPanel.style.display = 'none';

    startBenchmarkBtn.textContent = 'Start Benchmark';
    pollBenchmarkStatus();
  } catch (error) {
    showAlert(`Error starting benchmark: ${error.message}`, 'error');
    startBenchmarkBtn.disabled = false;
    startBenchmarkBtn.textContent = 'Start Benchmark';
  }
}

function pollBenchmarkStatus() {
  statusCheckInterval = setInterval(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/benchmark/status`);
      if (!response.ok) throw new Error('Failed to fetch status');

      const status = await response.json();
      updateProgressUI(status);

      if (!status.running) {
        clearInterval(statusCheckInterval);
        benchmarkRunning = false;
        loadResults();
      }
    } catch (error) {
      console.error('Error polling status:', error);
    }
  }, STATUS_POLL_INTERVAL);
}

function updateProgressUI(status) {
  const progress = status.total > 0 ? (status.progress / status.total) * 100 : 0;
  progressFill.style.width = `${progress}%`;
  progressText.textContent = `${status.progress} / ${status.total}`;
  currentProblem.textContent = status.current_problem || 'Initializing...';

  const elapsed = Math.floor((Date.now() - benchmarkStartTime) / 1000);
  timeElapsed.textContent = `Time elapsed: ${formatSeconds(elapsed)}`;
}

async function loadResults() {
  try {
    const response = await fetch(`${BACKEND_URL}/benchmark/results`);
    if (!response.ok) throw new Error('Failed to load results');

    currentResults = await response.json();
    displayResults(currentResults);

    statusPanel.style.display = 'none';
    resultsPanel.style.display = 'block';
    configPanel.style.display = 'block';
    startBenchmarkBtn.disabled = false;

    showAlert('Benchmark completed successfully!', 'success');
  } catch (error) {
    showAlert(`Error loading results: ${error.message}`, 'error');
  }
}

function displayResults(data) {
  const summary = data.summary;

  // Overall Metrics
  overallMetrics.innerHTML = '';
  for (const [mode, metrics] of Object.entries(summary.overall)) {
    overallMetrics.appendChild(
      createMetricCard(
        mode.replace(/_/g, ' ').toUpperCase(),
        metrics.pass_rate,
        metrics.avg_latency_ms,
        metrics.textgrad_delta
      )
    );
  }

  // Difficulty Breakdown
  difficultyMetrics.innerHTML = '';
  for (const [difficulty, modes] of Object.entries(summary.by_difficulty)) {
    const container = document.createElement('div');
    container.className = 'difficulty-breakdown';
    const title = document.createElement('h4');
    title.textContent = difficulty.toUpperCase();
    container.appendChild(title);

    for (const [mode, metrics] of Object.entries(modes)) {
      container.appendChild(
        createSmallMetricCard(
          mode.replace(/_/g, ' '),
          metrics.pass_rate,
          metrics.total
        )
      );
    }
    difficultyMetrics.appendChild(container);
  }

  // Platform Breakdown
  platformMetrics.innerHTML = '';
  for (const [platform, modes] of Object.entries(summary.by_platform)) {
    const container = document.createElement('div');
    container.className = 'platform-breakdown';
    const title = document.createElement('h4');
    title.textContent = platform || 'Unknown';
    container.appendChild(title);

    for (const [mode, metrics] of Object.entries(modes)) {
      container.appendChild(
        createSmallMetricCard(
          mode.replace(/_/g, ' '),
          metrics.pass_rate,
          metrics.total
        )
      );
    }
    platformMetrics.appendChild(container);
  }

  // Error Breakdown
  errorBreakdown.innerHTML = '';
  for (const [mode, errors] of Object.entries(summary.by_error_type)) {
    const modeDiv = document.createElement('div');
    modeDiv.className = 'error-mode-section';

    const title = document.createElement('h4');
    title.textContent = mode.replace(/_/g, ' ').toUpperCase();
    modeDiv.appendChild(title);

    const table = document.createElement('table');
    table.className = 'error-table';

    const headerRow = table.insertRow();
    headerRow.innerHTML = '<th>Error Type</th><th>Count</th>';

    for (const [errorType, count] of Object.entries(errors)) {
      const row = table.insertRow();
      row.innerHTML = `<td>${errorType}</td><td>${count}</td>`;
    }

    modeDiv.appendChild(table);
    errorBreakdown.appendChild(modeDiv);
  }
}

function createMetricCard(label, passRate, latency, textgradDelta) {
  const card = document.createElement('div');
  card.className = 'metric-card';

  let deltaHTML = '';
  if (textgradDelta !== undefined && textgradDelta !== null) {
    const deltaStr = (textgradDelta * 100).toFixed(1);
    const deltaClass = textgradDelta >= 0 ? 'positive' : 'negative';
    deltaHTML = `<div class="metric-delta ${deltaClass}">Δ ${deltaStr}%</div>`;
  }

  card.innerHTML = `
    <div class="metric-label">${label}</div>
    <div class="metric-value">${(passRate * 100).toFixed(1)}%</div>
    <div class="metric-subtext">Pass Rate</div>
    ${deltaHTML}
    <div class="metric-latency">${latency.toFixed(0)}ms avg</div>
  `;

  return card;
}

function createSmallMetricCard(label, passRate, total) {
  const card = document.createElement('div');
  card.className = 'small-metric-card';
  card.innerHTML = `
    <div class="small-metric-label">${label}</div>
    <div class="small-metric-value">${(passRate * 100).toFixed(1)}%</div>
    <div class="small-metric-subtext">${total} problems</div>
  `;
  return card;
}

function exportResults() {
  if (!currentResults) {
    showAlert('No results to export', 'warning');
    return;
  }

  const dataStr = JSON.stringify(currentResults, null, 2);
  const blob = new Blob([dataStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `benchmark_results_${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

async function loadLatestResults() {
  try {
    const response = await fetch(`${BACKEND_URL}/benchmark/results`);
    if (!response.ok) throw new Error('Failed to load results');

    currentResults = await response.json();
    displayResults(currentResults);

    configPanel.style.display = 'block';
    statusPanel.style.display = 'none';
    resultsPanel.style.display = 'block';

    showAlert('Latest results loaded', 'success');
  } catch (error) {
    showAlert(`Error loading results: ${error.message}`, 'error');
  }
}

async function showResultsHistory() {
  try {
    const response = await fetch(`${BACKEND_URL}/benchmark/results/all`);
    if (!response.ok) throw new Error('Failed to load history');

    const allResults = await response.json();
    displayResultsHistory(allResults);
    historyModal.style.display = 'flex';
  } catch (error) {
    showAlert(`Error loading history: ${error.message}`, 'error');
  }
}

function displayResultsHistory(allResults) {
  historyList.innerHTML = '';

  if (allResults.length === 0) {
    historyList.innerHTML = '<p style="text-align: center; color: #999;">No results history available</p>';
    return;
  }

  for (const result of allResults) {
    const item = document.createElement('div');
    item.className = 'history-item';

    const timestamp = new Date(result.timestamp).toLocaleString();
    const totalProblems = result.summary.overall.baseline?.total_problems || 0;
    const baselineRate = (
      (result.summary.overall.baseline?.pass_rate || 0) * 100
    ).toFixed(1);

    item.innerHTML = `
      <div class="history-item-header">
        <div class="history-timestamp">${timestamp}</div>
        <div class="history-stats">${totalProblems} problems | Baseline: ${baselineRate}%</div>
      </div>
    `;

    item.addEventListener('click', () => {
      currentResults = result;
      displayResults(result);
      historyModal.style.display = 'none';
      configPanel.style.display = 'block';
      statusPanel.style.display = 'none';
      resultsPanel.style.display = 'block';
    });

    historyList.appendChild(item);
  }
}

function closeResultsHistory() {
  historyModal.style.display = 'none';
}

function formatSeconds(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`;
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  } else {
    return `${secs}s`;
  }
}

function showAlert(message, type = 'info') {
  const alert = document.createElement('div');
  alert.className = `alert alert-${type}`;
  alert.textContent = message;
  document.body.appendChild(alert);

  setTimeout(() => {
    alert.classList.add('show');
  }, 10);

  setTimeout(() => {
    alert.classList.remove('show');
    setTimeout(() => alert.remove(), 300);
  }, 3000);
}
