const modelSelect = document.getElementById('modelSelect');
const customModel = document.getElementById('customModel');
const temperatureInput = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperatureValue');
const maxOutputTokensInput = document.getElementById('maxOutputTokens');
const apiKeySection = document.getElementById('apiKeySection');
const apiKeyInput = document.getElementById('apiKey');
const includeRag = document.getElementById('includeRag');
const includeTextGrad = document.getElementById('includeTextGrad');
const textGradModelSection = document.getElementById('textGradModelSection');
const textGradModelSelect = document.getElementById('textGradModelSelect');
const customTextGradModel = document.getElementById('customTextGradModel');
const textGradApiKeySection = document.getElementById('textGradApiKeySection');
const textGradApiKeyInput = document.getElementById('textGradApiKey');
const textGradLoopsSection = document.getElementById('textGradLoopsSection');
const textGradLoopsInput = document.getElementById('textGradLoops');
const textGradInternalMaxOutputTokensSection = document.getElementById('textGradInternalMaxOutputTokensSection');
const textGradInternalMaxOutputTokensInput = document.getElementById('textGradInternalMaxOutputTokens');
const systemPrompt = document.getElementById('systemPrompt');
const textGradLossPromptSection = document.getElementById('textGradLossPromptSection');
const textGradLossPrompt = document.getElementById('textGradLossPrompt');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');
const downloadSettingsBtn = document.getElementById('downloadSettingsBtn');
const loadSettingsBtn = document.getElementById('loadSettingsBtn');
const settingsFileInput = document.getElementById('settingsFileInput');
const saveStatus = document.getElementById('saveStatus');
const backToChat = document.getElementById('backToChat');
const downloadDatasetBtn = document.getElementById('downloadDatasetBtn');
const datasetStatus = document.getElementById('datasetStatus');
const datasetVersionSelect = document.getElementById('datasetVersionSelect');
const buildRagBtn = document.getElementById('buildRagBtn');
const ragBuildStatus = document.getElementById('ragBuildStatus');
const datasetProgress = document.getElementById('datasetProgress');
const datasetProgressLabel = document.getElementById('datasetProgressLabel');
const datasetProgressPercent = document.getElementById('datasetProgressPercent');
const datasetProgressFill = document.getElementById('datasetProgressFill');
const ragProgress = document.getElementById('ragProgress');
const ragProgressLabel = document.getElementById('ragProgressLabel');
const ragProgressPercent = document.getElementById('ragProgressPercent');
const ragProgressFill = document.getElementById('ragProgressFill');
const ragProgressLog = document.getElementById('ragProgressLog');
const ollamaStatus = document.getElementById('ollamaStatus');
const refreshOllamaBtn = document.getElementById('refreshOllamaBtn');
const BACKEND_URL = 'http://localhost:5050';
const CUSTOM_CLOUD_MODEL = '__ollama_cloud__';
let datasetStatusInterval = null;
let ragBuildStatusInterval = null;

let defaultSettings = {};
let models = [];
let modelProviders = {};

function setProgress(container, fill, percentNode, labelNode, percent, label) {
  const safePercent = Math.max(0, Math.min(100, Number(percent) || 0));
  container.hidden = false;
  fill.style.width = `${safePercent}%`;
  percentNode.textContent = `${safePercent}%`;
  labelNode.textContent = label;
  fill.parentElement.setAttribute('aria-valuenow', safePercent);
}

function formatBytes(bytes) {
  if (!bytes) return '0 MB';
  const units = ['B', 'KB', 'MB', 'GB'];
  const unit = Math.min(Math.floor(Math.log(bytes) / Math.log(1000)), units.length - 1);
  return `${(bytes / (1000 ** unit)).toFixed(unit >= 3 ? 2 : 0)} ${units[unit]}`;
}

function startPolling(kind) {
  if (kind === 'dataset' && !datasetStatusInterval) {
    datasetStatusInterval = setInterval(refreshDatasetStatus, 1000);
  }
  if (kind === 'rag' && !ragBuildStatusInterval) {
    ragBuildStatusInterval = setInterval(refreshRagBuildStatus, 1000);
  }
}

function stopPolling(kind) {
  if (kind === 'dataset') {
    clearInterval(datasetStatusInterval);
    datasetStatusInterval = null;
  } else {
    clearInterval(ragBuildStatusInterval);
    ragBuildStatusInterval = null;
  }
}

function populateSelect(id) {
  const select = document.getElementById(id);
  select.replaceChildren();
  models.forEach(m => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = m;
    select.appendChild(opt);
  });
  const cloudOption = document.createElement('option');
  cloudOption.value = CUSTOM_CLOUD_MODEL;
  cloudOption.textContent = 'Other Ollama Cloud model…';
  select.appendChild(cloudOption);
}

function selectedModel(select, customInput) {
  return select.value === CUSTOM_CLOUD_MODEL ? customInput.value.trim() : select.value;
}

function setModelSelection(select, customInput, model) {
  const isListed = models.includes(model);
  select.value = isListed ? model : CUSTOM_CLOUD_MODEL;
  customInput.hidden = isListed;
  customInput.value = isListed ? '' : model;
}

function loadSettings() {
  return window.SettingsStore.load(defaultSettings, models);
}

function saveSettings(settings, requireApiKeys = true) {
  return window.SettingsStore.save(settings, defaultSettings, models, { requireApiKeys });
}

async function fetchDefaultSettings() {
  const response = await fetch(`${BACKEND_URL}/api/defaults`);
  if (!response.ok) throw new Error('Unable to load schema defaults.');

  const {
    models: schemaModels,
    modelProviders: schemaModelProviders = {},
    ...schemaDefaults
  } = await response.json();
  if (!Array.isArray(schemaModels) || schemaModels.length === 0) {
    throw new Error('The backend did not provide any supported models.');
  }
  models = schemaModels;
  modelProviders = schemaModelProviders;
  defaultSettings = {
    ...schemaDefaults,
    modelProviders,
    // Optional Pydantic API-key defaults are null; form inputs use strings.
    apiKey: schemaDefaults.apiKey || '',
    textGradApiKey: schemaDefaults.textGradApiKey || '',
  };
}

function getSettingsFromForm() {
  return {
    model: selectedModel(modelSelect, customModel),
    temperature: parseFloat(temperatureInput.value),
    maxOutputTokens: parseInt(maxOutputTokensInput.value, 10),
    includeRag: includeRag.checked,
    includeTextGrad: includeTextGrad.checked,
    textGradModel: selectedModel(textGradModelSelect, customTextGradModel),
    textGradLoops: Math.min(5, Math.max(1, parseInt(textGradLoopsInput.value, 10) || 1)),
    textGradInternalMaxOutputTokens: parseInt(textGradInternalMaxOutputTokensInput.value, 10),
    textGradLossPrompt: textGradLossPrompt.value.trim() || 'Evaluate this answer. It should be factual, clear, and directly answer the question.',
    systemPrompt: systemPrompt.value,
    apiKey: apiKeyInput.value.trim(),
    textGradApiKey: textGradApiKeyInput.value.trim(),
  };
}

function validateImportedSettings(settings) {
  return window.SettingsStore.normalize(settings, defaultSettings, models, { requireApiKeys: true });
}

function updateSliderValue() {
  temperatureValue.textContent = parseFloat(temperatureInput.value).toFixed(1);
}

function populateForm(settings) {
  setModelSelection(modelSelect, customModel, settings.model || defaultSettings.model || models[0]);
  temperatureInput.value = settings.temperature ?? 0;
  maxOutputTokensInput.value = settings.maxOutputTokens ?? defaultSettings.maxOutputTokens;
  updateSliderValue();
  apiKeyInput.value = settings.apiKey || '';
  includeRag.checked = settings.includeRag ?? false;
  includeTextGrad.checked = settings.includeTextGrad ?? false;
  setModelSelection(
    textGradModelSelect,
    customTextGradModel,
    settings.textGradModel || defaultSettings.textGradModel || models[0],
  );
  textGradApiKeyInput.value = settings.textGradApiKey || '';
  textGradLoopsInput.value = Math.min(5, Math.max(1, settings.textGradLoops || 1));
  textGradInternalMaxOutputTokensInput.value = settings.textGradInternalMaxOutputTokens
    ?? defaultSettings.textGradInternalMaxOutputTokens;
  systemPrompt.value = settings.systemPrompt || '';
  textGradLossPrompt.value = settings.textGradLossPrompt || 'Evaluate this answer. It should be factual, clear, and directly answer the question.';
  updateApiKeyVisibility(selectedModel(modelSelect, customModel));
  updateTextGradApiKeyVisibility(selectedModel(textGradModelSelect, customTextGradModel), settings.includeTextGrad);
  updateTextGradVisibility(settings.includeTextGrad);
}

function showStatus(message) {
  saveStatus.textContent = message;
  saveStatus.style.opacity = '1';
  setTimeout(() => {
    saveStatus.style.opacity = '0';
  }, 1800);
}

function needsApiKey(model) {
  return modelProviders[model] !== 'ollama_local' && model !== 'mock-chat:1.0';
}

function needsTextGradApiKey(textGradModel, includeTextGrad) {
  return includeTextGrad && textGradModel && needsApiKey(textGradModel);
}

async function refreshOllamaStatus() {
  refreshOllamaBtn.disabled = true;
  ollamaStatus.classList.remove('status-success', 'status-error');
  ollamaStatus.textContent = 'Checking the local Ollama service...';
  try {
    const response = await fetch(`${BACKEND_URL}/api/ollama/status`);
    if (!response.ok) throw new Error(`Status request failed (${response.status})`);
    const status = await response.json();
    if (!status.running) {
      ollamaStatus.textContent = `Ollama is not running at ${status.host}. Start Ollama, then refresh.`;
      ollamaStatus.classList.add('status-error');
      return;
    }

    const primarySelection = selectedModel(modelSelect, customModel);
    const textGradSelection = selectedModel(textGradModelSelect, customTextGradModel);
    for (const model of status.models || []) {
      if (!models.includes(model)) models.push(model);
      modelProviders[model] = 'ollama_local';
    }
    defaultSettings.modelProviders = modelProviders;
    populateSelect('modelSelect');
    populateSelect('textGradModelSelect');
    setModelSelection(modelSelect, customModel, primarySelection || defaultSettings.model);
    setModelSelection(textGradModelSelect, customTextGradModel, textGradSelection || defaultSettings.textGradModel);
    updateApiKeyVisibility(selectedModel(modelSelect, customModel));
    updateTextGradApiKeyVisibility(selectedModel(textGradModelSelect, customTextGradModel), includeTextGrad.checked);

    const count = (status.models || []).length;
    ollamaStatus.textContent = count
      ? `Connected at ${status.host}. ${count} local model${count === 1 ? '' : 's'} available.`
      : `Connected at ${status.host}, but no models are installed. Run ollama pull <model>.`;
    ollamaStatus.classList.add(count ? 'status-success' : 'status-error');
  } catch (error) {
    ollamaStatus.textContent = 'Unable to query Ollama. Make sure the backend is running.';
    ollamaStatus.classList.add('status-error');
  } finally {
    refreshOllamaBtn.disabled = false;
  }
}

function updateApiKeyVisibility(model) {
  if (needsApiKey(model)) {
    apiKeySection.style.display = 'block';
    apiKeyInput.required = true;
  } else {
    apiKeySection.style.display = 'none';
    apiKeyInput.required = false;
  }
}

function updateTextGradApiKeyVisibility(textGradModel, includeTextGrad) {
  if (needsTextGradApiKey(textGradModel, includeTextGrad)) {
    textGradApiKeySection.style.display = 'block';
    textGradApiKeyInput.required = true;
  } else {
    textGradApiKeySection.style.display = 'none';
    textGradApiKeyInput.required = false;
  }
}

function updateTextGradVisibility(enabled) {
  textGradModelSection.style.display = enabled ? 'block' : 'none';
  updateTextGradApiKeyVisibility(textGradModelSelect.value, enabled);
  textGradLoopsSection.style.display = enabled ? 'block' : 'none';
  textGradInternalMaxOutputTokensSection.style.display = enabled ? 'block' : 'none';
  textGradLossPromptSection.style.display = enabled ? 'block' : 'none';
  textGradLoopsInput.required = enabled;
  textGradInternalMaxOutputTokensInput.required = enabled;
}

async function refreshRagBuildStatus() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/rag/build/status`);
    if (!response.ok) throw new Error('Unable to check RAG build status');
    const status = await response.json();
    if (status.running) {
      ragBuildStatus.textContent = status.message || 'Running the full RAG pipeline...';
      buildRagBtn.disabled = true;
      buildRagBtn.textContent = 'Generating RAG...';
      setProgress(ragProgress, ragProgressFill, ragProgressPercent, ragProgressLabel,
        status.percent, status.message || status.stage);
      if (status.output) {
        ragProgressLog.hidden = false;
        ragProgressLog.textContent = status.output;
        ragProgressLog.scrollTop = ragProgressLog.scrollHeight;
      }
      startPolling('rag');
      return;
    }
    stopPolling('rag');
    if (status.error) {
      setProgress(ragProgress, ragProgressFill, ragProgressPercent, ragProgressLabel,
        status.percent, `Failed: ${status.error}`);
    } else {
      ragProgress.hidden = true;
      ragProgressLog.hidden = true;
    }
    if (status.chunks_exist) {
      buildRagBtn.disabled = false;
      buildRagBtn.textContent = 'Rebuild RAG index';
      ragBuildStatus.textContent = `${status.chunk_count} RAG chunks are ready. Rebuilding replaces the existing index.`;
      return;
    }
    buildRagBtn.disabled = false;
    buildRagBtn.textContent = 'Create RAG chunks';
    ragBuildStatus.textContent = status.error
      ? `Chunk creation failed: ${status.error}`
      : status.output
        ? 'RAG chunks created and the search index is ready.'
        : 'Create chunks from the scraped RAG data and rebuild the local search index.';
  } catch (error) {
    ragBuildStatus.textContent = 'RAG build status unavailable. Is the backend running?';
    buildRagBtn.disabled = true;
  }
}

async function refreshDatasetStatus() {
  const version = datasetVersionSelect.value;
  try {
    const response = await fetch(`${BACKEND_URL}/benchmark/dataset/status?version=${encodeURIComponent(version)}`);
    if (!response.ok) throw new Error('Unable to check dataset status');
    const status = await response.json();
    if (status.available) {
      datasetStatus.textContent = 'Downloaded and ready for benchmarks.';
      downloadDatasetBtn.disabled = true;
      downloadDatasetBtn.textContent = 'Dataset downloaded';
      datasetProgress.hidden = true;
      stopPolling('dataset');
    } else if (status.downloading) {
      const sizeDetail = status.total_bytes
        ? ` ${formatBytes(status.downloaded_bytes)} / ${formatBytes(status.total_bytes)}`
        : '';
      datasetStatus.textContent = `${status.message || 'Downloading Parquet shards...'}${sizeDetail}`;
      downloadDatasetBtn.disabled = true;
      downloadDatasetBtn.textContent = 'Downloading...';
      setProgress(datasetProgress, datasetProgressFill, datasetProgressPercent, datasetProgressLabel,
        status.percent, status.message || 'Preparing download...');
      startPolling('dataset');
    } else {
      stopPolling('dataset');
      datasetProgress.hidden = !status.error;
      datasetStatus.textContent = status.error
        ? `Download failed: ${status.error}`
        : 'Not downloaded. Download it before running a benchmark.';
      downloadDatasetBtn.disabled = false;
      downloadDatasetBtn.textContent = 'Download dataset';
    }
  } catch (error) {
    datasetStatus.textContent = 'Dataset status unavailable. Is the backend running?';
    downloadDatasetBtn.disabled = true;
  }
}

downloadDatasetBtn.addEventListener('click', async () => {
  downloadDatasetBtn.disabled = true;
  downloadDatasetBtn.textContent = 'Starting download...';
  try {
    const version = datasetVersionSelect.value;
    const response = await fetch(`${BACKEND_URL}/benchmark/dataset/download?version=${encodeURIComponent(version)}`, { method: 'POST' });
    if (!response.ok) throw new Error('Could not start the download');
    await refreshDatasetStatus();
    startPolling('dataset');
  } catch (error) {
    datasetStatus.textContent = error.message;
    downloadDatasetBtn.disabled = false;
    downloadDatasetBtn.textContent = 'Download dataset';
  }
});

datasetVersionSelect.addEventListener('change', () => {
  stopPolling('dataset');
  datasetProgress.hidden = true;
  refreshDatasetStatus();
});

buildRagBtn.addEventListener('click', async () => {
  buildRagBtn.disabled = true;
  buildRagBtn.textContent = 'Starting...';
  try {
    const response = await fetch(`${BACKEND_URL}/api/rag/build`, { method: 'POST' });
    if (!response.ok) throw new Error((await response.json()).detail || 'Could not start RAG chunk creation');
    await refreshRagBuildStatus();
    startPolling('rag');
  } catch (error) {
    ragBuildStatus.textContent = error.message;
    await refreshRagBuildStatus();
  }
});

temperatureInput.addEventListener('input', updateSliderValue);
modelSelect.addEventListener('change', () => {
  customModel.hidden = modelSelect.value !== CUSTOM_CLOUD_MODEL;
  updateApiKeyVisibility(selectedModel(modelSelect, customModel));
});
customModel.addEventListener('input', () => updateApiKeyVisibility(selectedModel(modelSelect, customModel)));
textGradModelSelect.addEventListener('change', () => {
  customTextGradModel.hidden = textGradModelSelect.value !== CUSTOM_CLOUD_MODEL;
  updateTextGradApiKeyVisibility(selectedModel(textGradModelSelect, customTextGradModel), includeTextGrad.checked);
});
customTextGradModel.addEventListener('input', () => updateTextGradApiKeyVisibility(
  selectedModel(textGradModelSelect, customTextGradModel), includeTextGrad.checked,
));
includeTextGrad.addEventListener('change', () => {
  updateTextGradVisibility(includeTextGrad.checked);
  updateTextGradApiKeyVisibility(textGradModelSelect.value, includeTextGrad.checked);
});

saveSettingsBtn.addEventListener('click', () => {
  try {
    saveSettings(getSettingsFromForm());
    showStatus('Settings saved locally.');
  } catch (error) {
    showStatus(`Could not save settings: ${error.message}`);
  }
});

downloadSettingsBtn.addEventListener('click', () => {
  try {
    const validatedSettings = window.SettingsStore.normalize(
      getSettingsFromForm(), defaultSettings, models, { requireApiKeys: true }
    );
    const exportFile = {
      format: 'fyp-chat-settings',
      version: 1,
      exportedAt: new Date().toISOString(),
      settings: validatedSettings,
    };
    const blob = new Blob([JSON.stringify(exportFile, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `fyp-settings-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
    showStatus('Settings downloaded. Keep the file secure.');
  } catch (error) {
    showStatus(`Could not download settings: ${error.message}`);
  }
});

loadSettingsBtn.addEventListener('click', () => settingsFileInput.click());
refreshOllamaBtn.addEventListener('click', refreshOllamaStatus);

settingsFileInput.addEventListener('change', async () => {
  const [file] = settingsFileInput.files;
  settingsFileInput.value = '';
  if (!file) return;

  try {
    const parsed = JSON.parse(await file.text());
    if (parsed?.format === 'fyp-chat-settings' && parsed.version !== 1) {
      throw new Error(`Unsupported settings file version: ${parsed.version}.`);
    }
    const importedSettings = validateImportedSettings(
      parsed?.format === 'fyp-chat-settings' ? parsed.settings : parsed
    );
    populateForm(importedSettings);
    saveSettings(importedSettings);
    showStatus('Settings loaded and saved locally.');
  } catch (error) {
    showStatus(`Could not load settings: ${error.message}`);
  }
});

document.addEventListener("DOMContentLoaded", async () => {
  try {
    await fetchDefaultSettings();
    populateSelect("modelSelect");
    populateSelect("textGradModelSelect");
  } catch (error) {
    showStatus('Could not load defaults from the backend.');
    return;
  }
  const settings = loadSettings();
  populateForm(settings);
  refreshOllamaStatus();
  refreshDatasetStatus();
  refreshRagBuildStatus();
  // ... rest of your existing DOMContentLoaded code
});

if (backToChat) {
  backToChat.addEventListener('click', () => {
    window.navigateWithTransition('index.html');
  });
}
