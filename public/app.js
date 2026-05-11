const chatWindow = document.getElementById('chatWindow');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const settingsButton = document.getElementById('settingsButton');
const newChatBtn = document.getElementById('newChatBtn');
const backendStatus = document.getElementById('backendStatus');

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

function loadSavedSettings() {
  const stored = localStorage.getItem('fyp_chat_settings');
  if (!stored) return defaultSettings;
  try {
    return { ...defaultSettings, ...JSON.parse(stored) };
  } catch (error) {
    console.error('Failed to parse saved settings', error);
    return defaultSettings;
  }
}

const state = {
  messages: [
    {
      role: 'assistant',
      text: 'Welcome! This demo UI is styled like ChatGPT. Open the settings page to customize model, benchmark, or evaluation settings.',
    },
  ],
  settings: loadSavedSettings(),
};

function formatMessage(text) {
  // Basic code block formatting
  return text
    .replace(/```(\w+)?\n?([\s\S]*?)```/g, '<pre><code class="language-$1">$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

function renderMessages() {
  chatWindow.innerHTML = '';
  state.messages.forEach((message) => {
    const row = document.createElement('div');
    row.className = `message ${message.role}`;
    const formattedText = formatMessage(message.text);
    row.innerHTML = `<strong>${message.role === 'assistant' ? 'Assistant' : 'You'}</strong><div>${formattedText}</div>`;
    chatWindow.appendChild(row);
  });
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addMessage(role, text) {
  state.messages.push({ role, text });
  renderMessages();
}

function updateTheme(darkMode) {
  document.documentElement.style.colorScheme = darkMode ? 'dark' : 'light';
  document.body.style.background = darkMode
    ? 'radial-gradient(circle at top left, rgba(16, 163, 127, 0.16), transparent 28%), radial-gradient(circle at 80% 10%, rgba(255, 255, 255, 0.08), transparent 18%), #0b1024'
    : 'radial-gradient(circle at top left, rgba(16, 163, 127, 0.12), transparent 28%), radial-gradient(circle at 80% 10%, rgba(15, 23, 42, 0.06), transparent 18%), #f8fafc';
  document.body.style.color = darkMode ? '#e5e7eb' : '#111827';
}

function saveSettings(settings) {
  localStorage.setItem('fyp_chat_settings', JSON.stringify(settings));
}

function setBackendStatus(connected) {
  if (!backendStatus) return;
  backendStatus.textContent = connected ? 'Backend: connected' : 'Backend: disconnected';
  backendStatus.style.color = connected ? '#a5f3fc' : '#fca5a5';
}

async function checkBackendStatus() {
  if (!backendStatus) return;
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('status ' + response.status);
    const data = await response.json();
    setBackendStatus(data.status === 'ok');
  } catch (error) {
    setBackendStatus(false);
    console.warn('Backend status check failed', error);
  }
}

async function sendMessageToServer(message) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      settings: state.settings,
    }),
  });

  if (!response.ok) {
    throw new Error('Server responded with ' + response.status);
  }

  const data = await response.json();
  return data.reply;
}

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;

  addMessage('user', text);
  messageInput.value = '';

  const loaderMessage = document.createElement('div');
  loaderMessage.className = 'message assistant';
  loaderMessage.innerHTML = `<strong>Assistant</strong><div>Typing...</div>`;
  chatWindow.appendChild(loaderMessage);
  chatWindow.scrollTop = chatWindow.scrollHeight;

  try {
    const reply = await sendMessageToServer(text);
    loaderMessage.remove();
    addMessage('assistant', reply);
  } catch (error) {
    loaderMessage.remove();
    addMessage('assistant', 'Server not available. Start the backend and reload the page.');
    console.error(error);
  }
});

if (settingsButton) {
  settingsButton.addEventListener('click', () => {
    window.location.href = 'settings.html';
  });
}

if (newChatBtn) {
  newChatBtn.addEventListener('click', () => {
    window.location.href = 'new-chat.html';
  });
}

renderMessages();
updateTheme(state.settings.darkTheme);
checkBackendStatus();
if (backendStatus) {
  setInterval(checkBackendStatus, 5000);
}
