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
  systemPrompt: `You are an expert competitive programmer. Given a competitive programming 
problem, produce a correct and efficient solution.

Your response must follow this exact structure:
1. APPROACH: Brief explanation of your algorithm and why it's correct
2. COMPLEXITY: Time and space complexity analysis
3. CODE: Complete, runnable solution in Python (or C++ if specified)

Requirements:
- Handle all edge cases explicitly
- Ensure your solution fits within the given time and memory constraints
- Output only the final solution code block, no partial attempts
- Do not include test scaffolding or input parsing beyond what is needed`,
  darkTheme: true,
  temperature: 0,
  benchmark: 'TruthfulQA',
  includeRag: false,
  includeTextGrad: true,
  textGradModel: 'gemma3:4b',
  textGradLoops: 1,
  textGradLossPrompt: `You are evaluating a competitive programming solution. Your feedback will 
be used to improve the prompt that generated this solution.

Evaluate the solution on these criteria:
1. CORRECTNESS: Does the logic handle all cases including edge cases?
2. COMPLEXITY: Is the time/space complexity optimal for the constraints?
3. COMPLETENESS: Is the solution fully implemented and runnable?
4. CLARITY: Is the approach clearly explained?

For each criterion, state:
- What the solution did well
- What specific weakness exists
- How the PROMPT (not the code) should be changed to elicit a better solution

Focus your feedback on prompt-level issues — e.g. "the prompt should instruct 
the model to explicitly consider overflow", not "the code has a bug on line 5". 
The goal is to improve the instruction, not patch the output directly.`,
  apiKey: '',
  textGradApiKey: '',
};
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

document.addEventListener("DOMContentLoaded", () => {
  populateSelect("modelSelect");
  populateSelect("textGradModelSelect");
  const settings = loadSettings();
  populateForm(settings);
  // ... rest of your existing DOMContentLoaded code
});

if (backToChat) {
  backToChat.addEventListener('click', () => {
    window.location.href = 'index.html';
  });
}
