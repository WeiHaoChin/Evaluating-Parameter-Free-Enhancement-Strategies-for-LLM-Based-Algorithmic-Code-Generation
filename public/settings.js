const modelSelect = document.getElementById('modelSelect');
const temperatureInput = document.getElementById('temperature');
const temperatureValue = document.getElementById('temperatureValue');
const apiKeySection = document.getElementById('apiKeySection');
const apiKeyInput = document.getElementById('apiKey');
const includeRag = document.getElementById('includeRag');
const includeTextGrad = document.getElementById('includeTextGrad');
const textGradModelSection = document.getElementById('textGradModelSection');
const textGradModelSelect = document.getElementById('textGradModelSelect');
const textGradApiKeySection = document.getElementById('textGradApiKeySection');
const textGradApiKeyInput = document.getElementById('textGradApiKey');
const textGradLoopsSection = document.getElementById('textGradLoopsSection');
const textGradLoopsInput = document.getElementById('textGradLoops');
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
const BACKEND_URL = 'http://localhost:5050';
let datasetStatusInterval = null;
let ragBuildStatusInterval = null;

let defaultSettings = {};
let models = [];

function populateSelect(id) {
  const select = document.getElementById(id);
  select.replaceChildren();
  models.forEach(m => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = m;
    select.appendChild(opt);
  });
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

  const { models: schemaModels, ...schemaDefaults } = await response.json();
  if (!Array.isArray(schemaModels) || schemaModels.length === 0) {
    throw new Error('The backend did not provide any supported models.');
  }
  models = schemaModels;
  defaultSettings = {
    ...schemaDefaults,
    // Optional Pydantic API-key defaults are null; form inputs use strings.
    apiKey: schemaDefaults.apiKey || '',
    textGradApiKey: schemaDefaults.textGradApiKey || '',
  };
}

function getSettingsFromForm() {
  return {
    model: modelSelect.value,
    temperature: parseFloat(temperatureInput.value),
    includeRag: includeRag.checked,
    includeTextGrad: includeTextGrad.checked,
    textGradModel: textGradModelSelect.value,
    textGradLoops: Math.min(5, Math.max(1, parseInt(textGradLoopsInput.value, 10) || 1)),
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
  temperatureValue.textContent = parseFloat(temperatureInput.value).toFixed(2);
}

function populateForm(settings) {
  modelSelect.value = models.includes(settings.model) ? settings.model : (defaultSettings.model || models[0]);
  temperatureInput.value = settings.temperature ?? 0;
  updateSliderValue();
  apiKeyInput.value = settings.apiKey || '';
  includeRag.checked = settings.includeRag ?? false;
  includeTextGrad.checked = settings.includeTextGrad ?? false;
  textGradModelSelect.value = models.includes(settings.textGradModel)
    ? settings.textGradModel
    : (defaultSettings.textGradModel || models[0]);
  textGradApiKeyInput.value = settings.textGradApiKey || '';
  textGradLoopsInput.value = Math.min(5, Math.max(1, settings.textGradLoops || 1));
  systemPrompt.value = settings.systemPrompt || '';
  textGradLossPrompt.value = settings.textGradLossPrompt || 'Evaluate this answer. It should be factual, clear, and directly answer the question.';
  updateApiKeyVisibility(settings.model);
  updateTextGradApiKeyVisibility(settings.textGradModel, settings.includeTextGrad);
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
  return model !== 'mock-chat:1.0';
}

function needsTextGradApiKey(textGradModel, includeTextGrad) {
  return includeTextGrad && textGradModel && textGradModel !== 'mock-chat:1.0';
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
  textGradLossPromptSection.style.display = enabled ? 'block' : 'none';
  textGradLoopsInput.required = enabled;
}

async function refreshRagBuildStatus() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/rag/build/status`);
    if (!response.ok) throw new Error('Unable to check RAG build status');
    const status = await response.json();
    if (status.running) {
      ragBuildStatus.textContent = 'Creating RAG chunks and rebuilding the search index. This may take several minutes...';
      buildRagBtn.disabled = true;
      buildRagBtn.textContent = 'Creating chunks...';
      return;
    }
    clearInterval(ragBuildStatusInterval);
    if (status.chunks_exist) {
      buildRagBtn.disabled = true;
      buildRagBtn.textContent = 'RAG chunks already exist';
      ragBuildStatus.textContent = `${status.chunk_count} RAG chunks already exist and are ready to use.`;
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
      clearInterval(datasetStatusInterval);
    } else if (status.downloading) {
      datasetStatus.textContent = 'Downloading from Hugging Face. This may take a few minutes...';
      downloadDatasetBtn.disabled = true;
      downloadDatasetBtn.textContent = 'Downloading...';
    } else {
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
    clearInterval(datasetStatusInterval);
    datasetStatusInterval = setInterval(refreshDatasetStatus, 2000);
  } catch (error) {
    datasetStatus.textContent = error.message;
    downloadDatasetBtn.disabled = false;
    downloadDatasetBtn.textContent = 'Download dataset';
  }
});

datasetVersionSelect.addEventListener('change', refreshDatasetStatus);

buildRagBtn.addEventListener('click', async () => {
  buildRagBtn.disabled = true;
  buildRagBtn.textContent = 'Starting...';
  try {
    const response = await fetch(`${BACKEND_URL}/api/rag/build`, { method: 'POST' });
    if (!response.ok) throw new Error((await response.json()).detail || 'Could not start RAG chunk creation');
    await refreshRagBuildStatus();
    clearInterval(ragBuildStatusInterval);
    ragBuildStatusInterval = setInterval(refreshRagBuildStatus, 2000);
  } catch (error) {
    ragBuildStatus.textContent = error.message;
    await refreshRagBuildStatus();
  }
});

temperatureInput.addEventListener('input', updateSliderValue);
modelSelect.addEventListener('change', () => updateApiKeyVisibility(modelSelect.value));
textGradModelSelect.addEventListener('change', () => updateTextGradApiKeyVisibility(textGradModelSelect.value, includeTextGrad.checked));
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
  refreshDatasetStatus();
  refreshRagBuildStatus();
  // ... rest of your existing DOMContentLoaded code
});

if (backToChat) {
  backToChat.addEventListener('click', () => {
    window.navigateWithTransition('index.html');
  });
}
