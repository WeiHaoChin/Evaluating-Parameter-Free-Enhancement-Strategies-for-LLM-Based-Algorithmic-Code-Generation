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
const startQuestion = document.getElementById('startQuestion');
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
  backendStatus.classList.remove('connected', 'error');
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
  const startAt = Number(startQuestion.value);
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
  if (!Number.isSafeInteger(startAt) || startAt < 1 || startAt > n) {
    showAlert(`Start question must be between 1 and ${n}`, 'error');
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
        startQuestion: startAt,
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
        metrics.textgrad_delta,
        metrics.average_output_tokens,
        metrics.average_model_wall_time_ms,
        metrics.average_server_total_duration,
        metrics.macro_evaluator_group_accuracy,
        metrics.evaluator_groups_passed,
        metrics.evaluator_groups_total
      )
    );
  }

  displayPairedWorkflow(summary, data.results || []);

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
  const errorTypesByMode = summary.by_error_type || {};
  const preferredModes = ['baseline', 'rag_only', 'textgrad_only', 'full'];
  const modes = [
    ...preferredModes.filter((mode) => mode in errorTypesByMode),
    ...Object.keys(errorTypesByMode).filter((mode) => !preferredModes.includes(mode)),
  ];
  const preferredOutcomes = [
    'PASSED',
    'WRONG_ANSWER',
    'TIME_LIMIT_EXCEEDED',
    'RUNTIME_ERROR',
    'COMPILATION_ERROR',
  ];
  const availableOutcomes = new Set(
    modes.flatMap((mode) => Object.keys(errorTypesByMode[mode] || {}))
  );
  const outcomes = [
    ...preferredOutcomes.filter((outcome) => availableOutcomes.has(outcome)),
    ...[...availableOutcomes].filter((outcome) => !preferredOutcomes.includes(outcome)),
  ];

  if (modes.length > 0 && outcomes.length > 0) {
    const table = document.createElement('table');
    table.className = 'error-table';
    const headerRow = table.insertRow();
    ['Test Case Outcome', ...modes.map(formatModeLabel)].forEach((label) => {
      const header = document.createElement('th');
      header.textContent = label;
      headerRow.appendChild(header);
    });

    outcomes.forEach((outcome) => {
      const row = table.insertRow();
      const outcomeCell = row.insertCell();
      outcomeCell.textContent = outcome.replace(/_/g, ' ');
      modes.forEach((mode) => {
        const countCell = row.insertCell();
        countCell.textContent = errorTypesByMode[mode]?.[outcome] ?? 0;
      });
    });
    errorBreakdown.appendChild(table);
  }

  // Individual Problem Results
  displayProblemResults(data.results);
}

function displayPairedWorkflow(summary, results) {
  const container = document.getElementById('pairedWorkflow');
  if (!container) return;

  const overall = summary?.overall || {};
  const baseline = overall.baseline || {};
  const rag = overall.rag_only || {};
  const textgrad = overall.textgrad_only || {};
  const full = overall.full || {};
  const actualInput = Number(textgrad.total_input_tokens || 0)
    + Number(full.total_input_tokens || 0);
  const actualOutput = Number(textgrad.total_output_tokens || 0)
    + Number(full.total_output_tokens || 0);
  const reusedInput = Number(baseline.total_input_tokens || 0)
    + Number(rag.total_input_tokens || 0);
  const reusedOutput = Number(baseline.total_output_tokens || 0)
    + Number(rag.total_output_tokens || 0);
  const actualTotal = actualInput + actualOutput;
  const reusedTotal = reusedInput + reusedOutput;

  container.innerHTML = `
    <div class="paired-workflow-grid">
      <div class="paired-workflow-row">
        <span class="workflow-branch">No-RAG branch</span>
        <span class="workflow-step">Generate A inside TextGrad</span>
        <span class="workflow-arrow">&rarr;</span>
        <span class="workflow-step workflow-reused">Judge A as Baseline</span>
        <span class="workflow-arrow">&rarr;</span>
        <span class="workflow-step">Critique and regenerate A&prime;</span>
        <span class="workflow-arrow">&rarr;</span>
        <span class="workflow-step">Judge A&prime; as TextGrad</span>
      </div>
      <div class="paired-workflow-row">
        <span class="workflow-branch">RAG branch</span>
        <span class="workflow-step">Retrieve and generate B inside Full</span>
        <span class="workflow-arrow">&rarr;</span>
        <span class="workflow-step workflow-reused">Judge B as RAG</span>
        <span class="workflow-arrow">&rarr;</span>
        <span class="workflow-step">Critique and regenerate B&prime;</span>
        <span class="workflow-arrow">&rarr;</span>
        <span class="workflow-step">Judge B&prime; as Full</span>
      </div>
    </div>
    <p class="workflow-note">
      Baseline and RAG are paired views of initial generations, not additional
      generator calls. In particular, RAG reuses Full's initial RAG generation B.
    </p>
    <div class="token-accounting-grid">
      <div class="token-accounting-card">
        <span>Completed problems</span>
        <strong>${results.length.toLocaleString()}</strong>
      </div>
      <div class="token-accounting-card">
        <span>Actual unique input tokens</span>
        <strong>${actualInput.toLocaleString()}</strong>
      </div>
      <div class="token-accounting-card">
        <span>Actual unique output tokens</span>
        <strong>${actualOutput.toLocaleString()}</strong>
      </div>
      <div class="token-accounting-card">
        <span>Actual unique total tokens</span>
        <strong>${actualTotal.toLocaleString()}</strong>
      </div>
      <div class="token-accounting-card token-accounting-reused">
        <span>Reused tokens excluded from actual total</span>
        <strong>${reusedTotal.toLocaleString()}</strong>
      </div>
    </div>
    <p class="workflow-note">
      Per-mode totals remain useful as hypothetical workflow costs. Summing all
      four modes would double-count ${reusedTotal.toLocaleString()} tokens from
      the shared Baseline and RAG initial generations.
    </p>
  `;
}

function createMetricCard(
  label, passRate, latency, textgradDelta, outputTokens, modelWallTime,
  serverTotalDuration, macroGroupAccuracy, evaluatorGroupsPassed,
  evaluatorGroupsTotal
) {
  const card = document.createElement('div');
  card.className = 'metric-card';

  let deltaHTML = '';
  if (textgradDelta !== undefined && textgradDelta !== null) {
    const deltaStr = (textgradDelta * 100).toFixed(1);
    const deltaClass = textgradDelta >= 0 ? 'positive' : 'negative';
    deltaHTML = `<div class="metric-delta ${deltaClass}">Δ ${deltaStr}%</div>`;
  }
  const tokenHTML = Number.isFinite(Number(outputTokens))
    ? `<div class="metric-latency">${Number(outputTokens).toFixed(0)} output tokens/call avg</div>`
    : '<div class="metric-latency">Token usage unavailable</div>';
  const modelTimeHTML = Number.isFinite(Number(modelWallTime))
    ? `<div class="metric-latency">${Number(modelWallTime).toFixed(0)}ms client-call avg</div>`
    : '<div class="metric-latency">Client-call timing unavailable</div>';
  const serverTimeHTML = Number.isFinite(Number(serverTotalDuration))
    ? `<div class="metric-latency">${(Number(serverTotalDuration) / 1_000_000).toFixed(0)}ms Ollama server avg</div>`
    : '<div class="metric-latency">Server timing unavailable</div>';
  const macroAccuracyHTML = Number.isFinite(Number(macroGroupAccuracy))
    ? `<div class="metric-latency">${(Number(macroGroupAccuracy) * 100).toFixed(1)}% macro evaluator-group accuracy</div>`
    : '<div class="metric-latency">Macro evaluator-group accuracy unavailable</div>';
  const rawGroupsHTML = (
    Number.isFinite(Number(evaluatorGroupsPassed))
    && Number.isFinite(Number(evaluatorGroupsTotal))
  )
    ? `<div class="metric-latency">${Number(evaluatorGroupsPassed)}/${Number(evaluatorGroupsTotal)} raw evaluator groups passed</div>`
    : '';

  card.innerHTML = `
    <div class="metric-label">${label}</div>
    <div class="metric-value">${(passRate * 100).toFixed(1)}%</div>
    <div class="metric-subtext">Pass@1</div>
    ${deltaHTML}
    ${macroAccuracyHTML}
    ${rawGroupsHTML}
    <div class="metric-latency">${latency.toFixed(0)}ms end-to-end avg</div>
    ${modelTimeHTML}
    ${serverTimeHTML}
    ${tokenHTML}
  `;

  return card;
}

function createSmallMetricCard(label, passRate, total) {
  const card = document.createElement('div');
  card.className = 'small-metric-card';
  card.innerHTML = `
    <div class="small-metric-label">${label}</div>
    <div class="small-metric-value">${(passRate * 100).toFixed(1)}%</div>
    <div class="small-metric-subtext">Pass@1 across ${total} problems</div>
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
    <th>Baseline Test Cases</th>
    <th>RAG Only Test Cases</th>
    <th>TextGrad Only Test Cases</th>
    <th>RAG + TextGrad Test Cases</th>
  `;

  // Add result rows
  for (const result of results) {
    const problem = result.problem || {};
    const modes = result.modes || {};
    const modeNames = ['baseline', 'rag_only', 'textgrad_only', 'full'];
    const baselinePassRate = getPassRate(modes.baseline);

    const row = table.insertRow();
    row.className = modeNames.every((mode) => isPassAtOne(modes[mode]))
      ? 'row-all-passed'
      : 'row-mixed';

    const problemCell = row.insertCell();
    problemCell.textContent = problem.title || 'Untitled problem';
    problemCell.title = problem.title || '';

    modeNames.forEach((mode) => {
      const cell = row.insertCell();
      const passed = isPassAtOne(modes[mode]);
      cell.className = passed ? 'mode-cell mode-cell-passed' : 'mode-cell mode-cell-failed';
      cell.appendChild(createModePassRateContent(
        modes[mode],
        mode === 'baseline' ? null : baselinePassRate
      ));
    });
  }

  problemResultsContainer.appendChild(table);
}

function isPassAtOne(modeResult) {
  return modeResult?.passed === true;
}

function getPassRate(modeResult) {
  const rawPassRate = modeResult?.pass_rate;
  if (rawPassRate === null || rawPassRate === undefined) {
    return null;
  }
  const passRate = Number(rawPassRate);
  return Number.isFinite(passRate) ? passRate : null;
}

function createModePassRateContent(modeResult, baselinePassRate) {
  const container = document.createElement('div');
  container.className = 'mode-result-content';
  const value = document.createElement('span');
  value.className = 'mode-pass-rate';
  const passRate = getPassRate(modeResult);
  const passedTests = Number(modeResult?.passed_tests);
  const totalTests = Number(modeResult?.total_tests);
  if (
    passRate !== null
    && Number.isInteger(passedTests)
    && Number.isInteger(totalTests)
  ) {
    value.textContent = `${passedTests}/${totalTests} (${(passRate * 100).toFixed(1)}%)`;
  } else {
    value.textContent = passRate !== null ? `${(passRate * 100).toFixed(1)}%` : '—';
  }
  const outcomeCounts = modeResult?.test_outcome_counts;
  const failureDetails = outcomeCounts && typeof outcomeCounts === 'object'
    ? Object.entries(outcomeCounts)
      .filter(([outcome, count]) => outcome !== 'PASSED' && Number(count) > 0)
      .map(([outcome, count]) => `${formatOutcomeLabel(outcome)}: ${Number(count)}`)
    : [];
  if (failureDetails.length > 0) {
    if (
      passRate !== null
      && Number.isInteger(passedTests)
      && Number.isInteger(totalTests)
    ) {
      value.textContent = `${passedTests}/${totalTests} (${(passRate * 100).toFixed(1)}%; ${failureDetails.join(', ')})`;
    } else {
      value.textContent += ` (${failureDetails.join(', ')})`;
    }
  } else if (modeResult?.error_type) {
    value.textContent += ` (${formatOutcomeLabel(modeResult.error_type)})`;
  }
  container.appendChild(value);

  if (baselinePassRate !== null && passRate !== null) {
    const deltaPoints = (passRate - baselinePassRate) * 100;
    const comparison = document.createElement('span');
    comparison.className = deltaPoints > 0
      ? 'baseline-delta baseline-delta-increased'
      : deltaPoints < 0
        ? 'baseline-delta baseline-delta-decreased'
        : 'baseline-delta baseline-delta-unchanged';
    const indicator = deltaPoints > 0 ? '↑' : deltaPoints < 0 ? '↓' : '→';
    const sign = deltaPoints > 0 ? '+' : '';
    comparison.textContent = `${indicator} ${sign}${deltaPoints.toFixed(1)} pp vs baseline`;
    container.appendChild(comparison);
  }
  return container;
}

function formatModeLabel(mode) {
  const labels = {
    baseline: 'Baseline',
    rag_only: 'RAG Only',
    textgrad_only: 'TextGrad Only',
    full: 'RAG + TextGrad',
  };
  return labels[mode] || mode.replace(/_/g, ' ').toUpperCase();
}

function formatOutcomeLabel(outcome) {
  return outcome
    .toLowerCase()
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase());
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
