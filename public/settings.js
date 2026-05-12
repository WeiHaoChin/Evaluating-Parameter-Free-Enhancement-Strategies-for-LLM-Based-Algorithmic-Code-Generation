const modelSelect = document.getElementById('modelSelect');
const benchmarkSelect = document.getElementById('benchmarkSelect');
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
const darkThemeToggle = document.getElementById('darkThemeToggle');
const systemPrompt = document.getElementById('systemPrompt');
const textGradSystemPromptSection = document.getElementById('textGradSystemPromptSection');
const textGradLossPromptSection = document.getElementById('textGradLossPromptSection');
const textGradLossPrompt = document.getElementById('textGradLossPrompt');
const saveSettingsBtn = document.getElementById('saveSettingsBtn');
const saveStatus = document.getElementById('saveStatus');
const backToChat = document.getElementById('backToChat');

const defaultSettings = {
  model: 'gemma3:4b',
  systemPrompt: 'The explanation must be clear and beginner-friendly.',
  darkTheme: true,
  temperature: 0.7,
  benchmark: 'TruthfulQA',
  includeRag: true,
  includeTextGrad: true,
  textGradModel: 'gemma3:4b',
  textGradLoops: 1,
  textGradLossPrompt: 'Evaluate this answer. It should be factual, clear, and directly answer the question.',
  apiKey: '',
  textGradApiKey: '',
};

function loadSettings() {
  const stored = sessionStorage.getItem('fyp_chat_settings');
  if (!stored) return defaultSettings;
  try {
    return { ...defaultSettings, ...JSON.parse(stored) };
  } catch {
    return defaultSettings;
  }
}

function saveSettings(settings) {
  sessionStorage.setItem('fyp_chat_settings', JSON.stringify(settings));
}

function updateSliderValue() {
  temperatureValue.textContent = parseFloat(temperatureInput.value).toFixed(2);
}

function populateForm(settings) {
  modelSelect.value = settings.model;
  benchmarkSelect.value = settings.benchmark;
  temperatureInput.value = settings.temperature;
  updateSliderValue();
  apiKeyInput.value = settings.apiKey || '';
  includeRag.checked = settings.includeRag;
  includeTextGrad.checked = settings.includeTextGrad;
  textGradModelSelect.value = settings.textGradModel || 'gemma3:4b';
  textGradApiKeyInput.value = settings.textGradApiKey || '';
  textGradLoopsInput.value = settings.textGradLoops || 1;
  darkThemeToggle.checked = settings.darkTheme;
  systemPrompt.value = settings.systemPrompt;
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

const settings = loadSettings();
populateForm(settings);

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

  const newSettings = {
    model: modelSelect.value,
    benchmark: benchmarkSelect.value,
    temperature: parseFloat(temperatureInput.value),
    includeRag: includeRag.checked,
    includeTextGrad: includeTextGrad.checked,
    textGradModel: textGradModelSelect.value,
    textGradLoops: parseInt(textGradLoopsInput.value, 10) || 1,
    textGradLossPrompt: textGradLossPrompt.value.trim() || 'Evaluate this answer. It should be factual, clear, and directly answer the question.',
    darkTheme: darkThemeToggle.checked,
    systemPrompt: systemPrompt.value,
    apiKey: apiKeyInput.value.trim(),
    textGradApiKey: textGradApiKeyInput.value.trim(),
  };
  saveSettings(newSettings);
  showStatus('Settings saved locally.');
});

if (backToChat) {
  backToChat.addEventListener('click', () => {
    window.location.href = 'index.html';
  });
}
