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
const textGradSystemPromptSection = document.getElementById('textGradSystemPromptSection');
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
const BACKEND_URL = 'http://localhost:5050';
let datasetStatusInterval = null;

let defaultSettings = {};
const MODELS = [
  "gemma3:4b",
  "gpt-oss:120b",
  "gemini-2.5-pro",
  "qwen3-coder-next",
  "deepseek-v3.2",
  "claude-sonnet-4-6",
  "deepseek-v4-flash"
];

function populateSelect(id) {
  const select = document.getElementById(id);
  MODELS.forEach(m => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = m;
    select.appendChild(opt);
  });
}

function loadSettings() {
  const stored = sessionStorage.getItem('fyp_chat_settings');
  if (!stored) return defaultSettings;
  try {
    const { benchmark, darkTheme, ...savedSettings } = JSON.parse(stored);
    return { ...defaultSettings, ...savedSettings };
  } catch {
    return defaultSettings;
  }
}

function saveSettings(settings) {
  sessionStorage.setItem('fyp_chat_settings', JSON.stringify(settings));
}

async function fetchDefaultSettings() {
  const response = await fetch(`${BACKEND_URL}/api/defaults`);
  if (!response.ok) throw new Error('Unable to load schema defaults.');

  const schemaDefaults = await response.json();
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
    textGradLoops: parseInt(textGradLoopsInput.value, 10) || 1,
    textGradLossPrompt: textGradLossPrompt.value.trim() || 'Evaluate this answer. It should be factual, clear, and directly answer the question.',
    systemPrompt: systemPrompt.value,
    apiKey: apiKeyInput.value.trim(),
    textGradApiKey: textGradApiKeyInput.value.trim(),
  };
}

function validateImportedSettings(settings) {
  if (!settings || typeof settings !== 'object' || Array.isArray(settings)) {
    throw new Error('The selected file does not contain settings.');
  }

  const { benchmark, darkTheme, ...currentSettings } = settings;
  const imported = { ...defaultSettings, ...currentSettings };
  if (!MODELS.includes(imported.model) || !MODELS.includes(imported.textGradModel)) {
    throw new Error('The file contains a model that is not available in this version.');
  }
  if (!Number.isFinite(Number(imported.temperature)) || Number(imported.temperature) < 0 || Number(imported.temperature) > 1) {
    throw new Error('The file contains an invalid temperature.');
  }
  if (!Number.isInteger(Number(imported.textGradLoops)) || Number(imported.textGradLoops) < 1 || Number(imported.textGradLoops) > 50) {
    throw new Error('The file contains an invalid TextGrad loop count.');
  }
  for (const key of ['includeRag', 'includeTextGrad']) {
    if (typeof imported[key] !== 'boolean') throw new Error(`The file contains an invalid ${key} value.`);
  }
  for (const key of ['systemPrompt', 'textGradLossPrompt', 'apiKey', 'textGradApiKey']) {
    if (typeof imported[key] !== 'string') throw new Error(`The file contains an invalid ${key} value.`);
  }

  return {
    ...imported,
    temperature: Number(imported.temperature),
    textGradLoops: Number(imported.textGradLoops),
  };
}

function updateSliderValue() {
  temperatureValue.textContent = parseFloat(temperatureInput.value).toFixed(2);
}

function populateForm(settings) {
  modelSelect.value = settings.model || MODELS[0];
  temperatureInput.value = settings.temperature ?? 0;
  updateSliderValue();
  apiKeyInput.value = settings.apiKey || '';
  includeRag.checked = settings.includeRag ?? false;
  includeTextGrad.checked = settings.includeTextGrad ?? false;
  textGradModelSelect.value = settings.textGradModel || 'gemma3:4b';
  textGradApiKeyInput.value = settings.textGradApiKey || '';
  textGradLoopsInput.value = settings.textGradLoops || 1;
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
  textGradApiKeySection.style.display = enabled ? 'block' : 'none';
  textGradLoopsSection.style.display = enabled ? 'block' : 'none';
  textGradSystemPromptSection.style.display = enabled ? 'block' : 'none';
  textGradLossPromptSection.style.display = enabled ? 'block' : 'none';
  textGradLoopsInput.required = enabled;
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

temperatureInput.addEventListener('input', updateSliderValue);
modelSelect.addEventListener('change', () => updateApiKeyVisibility(modelSelect.value));
textGradModelSelect.addEventListener('change', () => updateTextGradApiKeyVisibility(textGradModelSelect.value, includeTextGrad.checked));
includeTextGrad.addEventListener('change', () => {
  updateTextGradVisibility(includeTextGrad.checked);
  updateTextGradApiKeyVisibility(textGradModelSelect.value, includeTextGrad.checked);
});

saveSettingsBtn.addEventListener('click', () => {
  if (needsApiKey(modelSelect.value) && !apiKeyInput.value.trim()) {
    showStatus('API key is required for the selected model.');
    return;
  }
  if (needsTextGradApiKey(textGradModelSelect.value, includeTextGrad.checked) && !textGradApiKeyInput.value.trim()) {
    showStatus('API key is required for the TextGrad model.');
    return;
  }

  const newSettings = getSettingsFromForm();
  saveSettings(newSettings);
  showStatus('Settings saved locally.');
});

downloadSettingsBtn.addEventListener('click', () => {
  const exportFile = {
    format: 'fyp-chat-settings',
    version: 1,
    exportedAt: new Date().toISOString(),
    settings: getSettingsFromForm(),
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
});

loadSettingsBtn.addEventListener('click', () => settingsFileInput.click());

settingsFileInput.addEventListener('change', async () => {
  const [file] = settingsFileInput.files;
  settingsFileInput.value = '';
  if (!file) return;

  try {
    const parsed = JSON.parse(await file.text());
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
  populateSelect("modelSelect");
  populateSelect("textGradModelSelect");
  try {
    await fetchDefaultSettings();
  } catch (error) {
    showStatus('Could not load defaults from the backend.');
  }
  const settings = loadSettings();
  populateForm(settings);
  refreshDatasetStatus();
  // ... rest of your existing DOMContentLoaded code
});

if (backToChat) {
  backToChat.addEventListener('click', () => {
    window.navigateWithTransition('index.html');
  });
}
