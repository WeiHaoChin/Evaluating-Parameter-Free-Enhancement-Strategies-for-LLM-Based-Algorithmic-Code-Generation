const BACKEND_URL = 'http://localhost:5050';
const STATUS_POLL_INTERVAL = 1000;

// DOM Elements
const configPanel = document.querySelector('.config-panel');
const statusPanel = document.getElementById('statusPanel');
const resultsPanel = document.getElementById('resultsPanel');
const startBenchmarkBtn = document.getElementById('startBenchmarkBtn');
const stopBenchmarkBtn = document.getElementById('stopBenchmarkBtn');
const versionSelect = document.getElementById('versionSelect');
const problemCount = document.getElementById('problemCount');
const problemCountSlider = document.getElementById('problemCountSlider');
const difficultySelect = document.getElementById('difficultySelect');
const randomSeed = document.getElementById('randomSeed');
const currentProblem = document.getElementById('currentProblem');
const progressText = document.getElementById('progressText');
const progressFill = document.getElementById('progressFill');
const timeElapsed = document.getElementById('timeElapsed');
const modeStatusCards = document.querySelectorAll('.mode-status-card');
const backendStatus = document.getElementById('backendStatus');
const overallMetrics = document.getElementById('overallMetrics');
const difficultyMetrics = document.getElementById('difficultyMetrics');
const platformMetrics = document.getElementById('platformMetrics');
const errorBreakdown = document.getElementById('errorBreakdown');
const exportResultsBtn = document.getElementById('exportResultsBtn');
const loadLatestResultsBtn = document.getElementById('loadLatestResultsBtn');
const viewResultsHistoryBtn = document.getElementById('viewResultsHistoryBtn');
const loadResultsFileBtn = document.getElementById('loadResultsFileBtn');
const resultsFileInput = document.getElementById('resultsFileInput');
const historyModal = document.getElementById('historyModal');
const closeHistoryBtn = document.getElementById('closeHistoryBtn');
const historyList = document.getElementById('historyList');
const problemResultsContainer = document.getElementById('problemResultsContainer');

let benchmarkRunning = false;
let benchmarkStartTime = null;
let statusCheckInterval = null;
let currentResults = null;
let defaultSettings = null;
let datasetAvailable = false;
let benchmarkReady = false;
let readiness = null;

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  linkSliderAndInput();
  checkBackendStatus();
  await fetchDefaultSettings();
  await refreshBenchmarkReadiness();
});

function setupEventListeners() {
  startBenchmarkBtn.addEventListener('click', startBenchmark);
  stopBenchmarkBtn.addEventListener('click', stopBenchmark);
  exportResultsBtn.addEventListener('click', exportResults);
  loadLatestResultsBtn.addEventListener('click', loadLatestResults);
  viewResultsHistoryBtn.addEventListener('click', showResultsHistory);
  loadResultsFileBtn.addEventListener('click', () => resultsFileInput.click());
  resultsFileInput.addEventListener('change', loadResultsFile);
  closeHistoryBtn.addEventListener('click', closeResultsHistory);
  historyModal.addEventListener('click', (e) => {
    if (e.target === historyModal) closeResultsHistory();
  });
  versionSelect.addEventListener('change', refreshBenchmarkReadiness);
  window.addEventListener('focus', refreshBenchmarkReadiness);
}

function renderReadiness(data) {
  const title = document.getElementById('readinessTitle');
  const count = document.getElementById('readinessCount');
  const checksContainer = document.getElementById('readinessChecks');
  const checks = Object.values(data.checks || {});
  const complete = checks.filter((check) => check.ready).length;

  title.textContent = data.ready ? 'Ready to run benchmark' : 'Complete setup before running';
  count.textContent = `${complete} / ${checks.length}`;
  count.classList.toggle('ready', data.ready);
  checksContainer.innerHTML = checks.map((check) => `
    <div class="readiness-check ${check.ready ? 'is-ready' : 'is-missing'}">
      <span class="readiness-icon" aria-hidden="true">${check.ready ? '&check;' : '!'}</span>
      <div><strong>${check.label}</strong><small>${check.ready ? 'Configured' : check.detail}</small></div>
    </div>
  `).join('');
}

function updateStartButton() {
  const canStart = datasetAvailable && benchmarkReady && !benchmarkRunning;
  startBenchmarkBtn.disabled = !canStart;
  if (canStart) {
    startBenchmarkBtn.textContent = 'Start Benchmark';
    startBenchmarkBtn.title = '';
  } else if (!datasetAvailable) {
    startBenchmarkBtn.textContent = 'Download dataset in Settings';
    startBenchmarkBtn.title = 'Download this dataset from Settings before running a benchmark.';
  } else {
    startBenchmarkBtn.textContent = 'Complete benchmark setup';
    startBenchmarkBtn.title = 'Configure RAG, the initial LLM, and TextGrad in Settings before running a benchmark.';
  }
}

async function refreshBenchmarkReadiness() {
  const version = versionSelect.value;
  try {
    const response = await fetch(`${BACKEND_URL}/benchmark/readiness`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ version, settings: loadSavedSettings() }),
      cache: 'no-store',
      signal: AbortSignal.timeout(10000),
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `Readiness check failed (${response.status})`);
    }
    readiness = await response.json();
    datasetAvailable = readiness.checks.dataset.ready;
    benchmarkReady = readiness.ready;
    renderReadiness(readiness);
    updateStartButton();
  } catch (error) {
    console.error('Benchmark readiness check failed:', error);
    datasetAvailable = false;
    benchmarkReady = false;
    startBenchmarkBtn.disabled = true;
    startBenchmarkBtn.textContent = 'Setup status unavailable';
    document.getElementById('readinessTitle').textContent = `Unable to check setup: ${error.message}`;
  }
}

function linkSliderAndInput() {
  problemCount.addEventListener('change', (e) => {
    problemCountSlider.value = e.target.value;
  });
  problemCountSlider.addEventListener('input', (e) => {
    problemCount.value = e.target.value;
  });
}

async function fetchDefaultSettings() {
  try {
    const response = await fetch(`${BACKEND_URL}/benchmark/defaults`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(10000),
    });
    if (response.ok) {
      defaultSettings = await response.json();
      console.log('Default settings loaded:', defaultSettings);
    } else {
      console.warn('Failed to fetch default settings, will use fallback');
      defaultSettings = null;
    }
  } catch (error) {
    console.error('Error fetching default settings:', error);
    defaultSettings = null;
  }
}

function loadSavedSettings() {
  const defaults = defaultSettings || {};
  return window.SettingsStore.load(defaults, defaults.models || []);
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

  if (!datasetAvailable) {
    showAlert('Download the selected LiveCodeBench dataset from Settings before running a benchmark.', 'warning');
    return;
  }

  if (!benchmarkReady) {
    showAlert('Complete the RAG and API-key requirements shown above before running a benchmark.', 'warning');
    return;
  }

  const n = parseInt(problemCount.value) || 10;
  const version = versionSelect.value;
  const difficulty = difficultySelect.value || null;
  const seed = Number(randomSeed.value);
  const settings = loadSavedSettings();
  console.log('Settings being sent:', settings);  // add this
  console.log('Model:', settings.model);           // add this

  if (n < 1 || n > 100) {
    showAlert('Problem count must be between 1 and 100', 'error');
    return;
  }
  if (!Number.isSafeInteger(seed)) {
    showAlert('Sampling seed must be an integer', 'error');
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
        seed,
        settings,
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
    await refreshBenchmarkReadiness();
  }
}

async function stopBenchmark() {
  if (!benchmarkRunning) {
    showAlert('No benchmark is currently running', 'warning');
    return;
  }

  try {
    stopBenchmarkBtn.disabled = true;
    stopBenchmarkBtn.textContent = 'Stopping...';

    const response = await fetch(`${BACKEND_URL}/benchmark/stop`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to stop benchmark');
    }

    showAlert('Benchmark stop requested. Please wait...', 'info');
  } catch (error) {
    showAlert(`Error stopping benchmark: ${error.message}`, 'error');
    stopBenchmarkBtn.disabled = false;
    stopBenchmarkBtn.textContent = 'Stop Benchmark';
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
        stopBenchmarkBtn.disabled = false;
        stopBenchmarkBtn.textContent = 'Stop Benchmark';
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

  modeStatusCards.forEach(card => {
    const mode = status.modes?.[card.dataset.mode] || {
      state: 'pending', detail: status.running ? 'Waiting to start' : 'Not running'
    };
    const badge = card.querySelector('.mode-status-badge');
    const detail = card.querySelector('.mode-status-detail');
    const labels = {
      pending: 'Waiting', retrieving: 'Retrieving', generating: 'Generating',
      getting_feedback: 'Feedback', optimizing: 'Optimizing', judging: 'Judging',
      complete: 'Complete', error: 'Failed'
    };
    card.dataset.state = mode.state || 'pending';
    badge.textContent = labels[mode.state] || mode.state || 'Waiting';
    detail.textContent = mode.detail || '';
  });
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
    await refreshBenchmarkReadiness();

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

  // Individual Problem Results
  displayProblemResults(data.results);
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

function displayProblemResults(results) {
  problemResultsContainer.innerHTML = '';

  if (!results || results.length === 0) {
    problemResultsContainer.innerHTML = '<p style="text-align: center; color: #999;">No problem results available</p>';
    return;
  }

  const table = document.createElement('table');
  table.className = 'problem-results-table';

  // Create header row
  const headerRow = table.insertRow();
  headerRow.className = 'header-row';
  headerRow.innerHTML = `
    <th>Problem</th>
    <th>Baseline Pass Rate</th>
    <th>RAG Only Pass Rate</th>
    <th>TextGrad Only Pass Rate</th>
    <th>RAG + TextGrad Pass Rate</th>
  `;

  // Add result rows
  for (const result of results) {
    const problem = result.problem || {};
    const modeResult = (result.modes || {}).baseline || {};

    const row = table.insertRow();
    row.className = modeResult.passed ? 'row-passed' : 'row-failed';

    const passedStatus = modeResult.passed ? '✓' : '✗';
    const passRatePercent = (modeResult.pass_rate * 100).toFixed(1);
    const textGradStatus = modeResult.textgrad_included ? 'Yes' : 'No';
    const ragContextStatus = modeResult.rag_context_included ? 'Yes' : 'No';
    const errorType = modeResult.error_type || '-';

    row.innerHTML = `
      <td title="${problem.title}">${problem.title}</td>
      <td>${formatModePassRate(result.modes?.baseline)}</td>
      <td>${formatModePassRate(result.modes?.rag_only)}</td>
      <td>${formatModePassRate(result.modes?.textgrad_only)}</td>
      <td>${formatModePassRate(result.modes?.full)}</td>
    `;
  }

  problemResultsContainer.appendChild(table);
}

function formatModePassRate(modeResult) {
  const passRate = Number(modeResult?.pass_rate);
  return Number.isFinite(passRate) ? `${(passRate * 100).toFixed(1)}%` : '—';
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

async function loadResultsFile(event) {
  const [file] = event.target.files;
  if (!file) return;

  try {
    const data = normaliseImportedResults(JSON.parse(await file.text()));
    currentResults = data;
    displayResults(data);

    configPanel.style.display = 'block';
    statusPanel.style.display = 'none';
    resultsPanel.style.display = 'block';
    showAlert(`Loaded results from ${file.name}`, 'success');
  } catch (error) {
    showAlert(`Unable to load results file: ${error.message}`, 'error');
  } finally {
    // Permit selecting the same file again after correcting or reviewing it.
    event.target.value = '';
  }
}

function normaliseImportedResults(data) {
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    throw new Error('the file does not contain a benchmark-results object');
  }
  if (!Array.isArray(data.results) || !data.summary || typeof data.summary !== 'object') {
    throw new Error('the file is missing benchmark results or its summary');
  }

  // Result exports made before the TextGrad flag was saved can still be viewed.
  for (const problemResult of data.results) {
    for (const [modeName, modeResult] of Object.entries(problemResult?.modes || {})) {
      if (modeResult && typeof modeResult === 'object' && modeResult.textgrad_included === undefined) {
        modeResult.textgrad_included = modeName === 'textgrad_only' || modeName === 'full';
      }
    }
  }
  return data;
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
    const totalProblems = result.summary?.overall?.baseline?.total_problems || 0;
    const baselineRate = (
      (result.summary?.overall?.baseline?.pass_rate || 0) * 100
    ).toFixed(1);
    const textGradSetting = result.settings?.includeTextGrad;
    const textGradLabel = textGradSetting === undefined || textGradSetting === null
      ? 'TextGrad setting unavailable'
      : `TextGrad enabled: ${textGradSetting ? 'Yes' : 'No'}`;

    item.innerHTML = `
      <div class="history-item-header">
        <div class="history-timestamp">${timestamp}</div>
        <div class="history-stats">${totalProblems} problems | Baseline: ${baselineRate}% | ${textGradLabel}</div>
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
