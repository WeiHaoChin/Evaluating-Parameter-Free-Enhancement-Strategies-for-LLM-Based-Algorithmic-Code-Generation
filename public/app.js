const chatWindow = document.getElementById('chatWindow');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const settingsButton = document.getElementById('settingsButton');
const newChatBtn = document.getElementById('newChatBtn');
const backendStatus = document.getElementById('backendStatus');

// WebSocket connection state
let ws = null;
let wsConnected = false;

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
  const stored = sessionStorage.getItem('fyp_chat_settings');
  if (!stored) return defaultSettings;
  try {
    return { ...defaultSettings, ...JSON.parse(stored) };
  } catch (error) {
    console.error('Failed to parse saved settings', error);
    return defaultSettings;
  }
}

function loadSavedMessages() {
  const stored = localStorage.getItem('fyp_chat_messages');
  if (!stored) return [];
  try {
    return JSON.parse(stored);
  } catch (error) {
    console.error('Failed to parse saved messages', error);
    return [];
  }
}

function saveMessages(messages) {
  try {
    localStorage.setItem('fyp_chat_messages', JSON.stringify(messages));
  } catch (error) {
    console.error('Failed to save messages', error);
  }
}

const state = {
  messages: loadSavedMessages().length > 0 ? loadSavedMessages() : [
    {
      role: 'assistant',
      text: 'Welcome! This demo UI is styled like ChatGPT. Open the settings page to customize model, benchmark, or evaluation settings.',
    },
  ],
  settings: loadSavedSettings(),
  currentAssistantMessage: null,
  currentTextGradDetails: null,
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
  saveMessages(state.messages);
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
  sessionStorage.setItem('fyp_chat_settings', JSON.stringify(settings));
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

// WebSocket Management Functions
function initializeWebSocket() {
  if (ws && wsConnected) return;
  
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/chat`;
  
  ws = new WebSocket(wsUrl);
  
  ws.onopen = () => {
    console.log('WebSocket connected');
    wsConnected = true;
  };
  
  ws.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      handleStreamingEvent(message);
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    wsConnected = false;
  };
  
  ws.onclose = () => {
    console.log('WebSocket closed');
    wsConnected = false;
  };
}

function sendMessageWebSocket(message) {
  if (!ws || !wsConnected) {
    throw new Error('WebSocket not connected');
  }
  ws.send(JSON.stringify({
    message,
    settings: state.settings,
  }));
}

function handleStreamingEvent(event) {
  const { type, data, loop, answer, original_prompt, updated, original } = event;
  
  switch (type) {
    case 'start':
      console.log('Processing started');
      break;
      
    case 'iteration_start':
      appendIterationStart(loop, original_prompt);
      break;
      
    case 'llm_response':
      appendLLMResponse(loop, data);
      break;
      
    case 'critic_feedback':
      appendCriticFeedback(loop, data);
      break;
      
    case 'prompt_updated':
      appendPromptUpdate(loop, original, updated);
      break;
      
    case 'iteration_complete':
      appendIterationComplete(loop);
      break;
      
    case 'complete':
      completeMessage(answer);
      break;
      
    case 'error':
      handleError(data);
      break;
      
    default:
      console.warn('Unknown event type:', type);
  }
}

function appendIterationStart(loop, originalPrompt) {
  const contentDiv = getOrCreateAssistantMessageContent();
  
  const iterationContainer = document.createElement('div');
  iterationContainer.className = 'iteration-container';
  iterationContainer.dataset.loop = loop;
  
  const toggleButton = document.createElement('button');
  toggleButton.className = 'iteration-toggle';
  toggleButton.innerHTML = `<span class="chevron">▼</span> Iteration ${loop}`;
  toggleButton.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    iterationContainer.classList.toggle('expanded');
  });
  
  const content = document.createElement('div');
  content.className = 'iteration-content';
  
  const details = document.createElement('div');
  details.className = 'iteration-details';
    
  if (originalPrompt) {
    const promptSection = document.createElement('div');
    promptSection.className = 'prompt-section';
    
    const label = document.createElement('strong');
    label.textContent = 'Original Prompt:';
    
    const promptEl = document.createElement('div');
    promptEl.id = `original-prompt-${loop}`;
    promptEl.className = 'streaming-text markdown-body';
    promptEl.innerHTML = marked.parse(originalPrompt);
    
    promptSection.appendChild(label);
    promptSection.appendChild(promptEl);
    details.appendChild(promptSection);
  }
  
  content.appendChild(details);
  iterationContainer.appendChild(toggleButton);
  iterationContainer.appendChild(content);
  contentDiv.appendChild(iterationContainer);
  
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendLLMResponse(loop, text) {
  const iterationContainer = findIterationContainer(loop);
  if (!iterationContainer) return;
  
  const details = iterationContainer.querySelector('.iteration-details');
  
  let responseSection = details.querySelector('[data-section="llm-response"]');
  if (!responseSection) {
    responseSection = document.createElement('div');
    responseSection.className = 'response-section';
    responseSection.dataset.section = 'llm-response';
    responseSection.innerHTML = `
      <strong>LLM Response:</strong>
      <div id="llm-response-${loop}" class="streaming-text markdown-body"></div>`;
    details.appendChild(responseSection);
  }
  
  const codeEl = document.getElementById(`llm-response-${loop}`);
  codeEl._raw = (codeEl._raw || '') + text;           // accumulate raw text
  codeEl.innerHTML = marked.parse(codeEl._raw);       // re-render as markdown
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendCriticFeedback(loop, text) {
  const iterationContainer = findIterationContainer(loop);
  if (!iterationContainer) return;
  
  const details = iterationContainer.querySelector('.iteration-details');
  
  let feedbackSection = details.querySelector('[data-section="critic-feedback"]');
  if (!feedbackSection) {
    feedbackSection = document.createElement('div');
    feedbackSection.className = 'feedback-section';
    feedbackSection.dataset.section = 'critic-feedback';
    feedbackSection.innerHTML = `
      <strong>Critic Feedback:</strong>
      <div id="critic-feedback-${loop}" class="streaming-text markdown-body"></div>`;
    details.appendChild(feedbackSection);
  }
  
  const codeEl = document.getElementById(`critic-feedback-${loop}`);
  codeEl._raw = (codeEl._raw || '') + text;
  codeEl.innerHTML = marked.parse(codeEl._raw);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendPromptUpdate(loop, original, updated) {
  const iterationContainer = findIterationContainer(loop);
  if (!iterationContainer) return;
  
  const details = iterationContainer.querySelector('.iteration-details');
  
  let updateSection = details.querySelector('[data-section="prompt-update"]');
  if (!updateSection) {
    updateSection = document.createElement('div');
    updateSection.className = 'prompt-update-section';
    updateSection.dataset.section = 'prompt-update';
    updateSection.innerHTML = `<strong>Updated Prompt:</strong><div id="updated-prompt-${loop}" class="streaming-text markdown-body"></div>`;
    details.appendChild(updateSection);
  }
  
  const promptEl = document.getElementById(`updated-prompt-${loop}`);
  if (updated) {
    promptEl.innerHTML = marked.parse(updated);
  }
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendIterationComplete(loop) {
  console.log(`Iteration ${loop} complete`);
}

function completeMessage(answer) {
  // Create assistant message if it doesn't exist
  if (!state.currentAssistantMessage) {
    const row = document.createElement('div');
    row.className = 'message assistant streaming';
    row.innerHTML = `<strong>Assistant</strong><div class="message-content"></div>`;
    chatWindow.appendChild(row);
    state.currentAssistantMessage = row;
  }
  
  // Add final answer at the top
  const contentDiv = state.currentAssistantMessage.querySelector('div');
  const finalSection = document.createElement('div');
  finalSection.className = 'final-answer-section';
  finalSection.innerHTML = `<strong>Final Answer:</strong><div class="markdown-body"></div><hr>`;
  const answerDiv = finalSection.querySelector('.markdown-body');
  answerDiv.innerHTML = marked.parse(answer);
  contentDiv.insertBefore(finalSection, contentDiv.firstChild);
  
  // Expand all iterations by default after completion
  const iterations = contentDiv.querySelectorAll('.iteration-container');
  iterations.forEach(iter => {
    if (!iter.classList.contains('expanded')) {
      iter.classList.add('expanded');
    }
  });
  
  state.currentAssistantMessage = null;
  saveMessages(state.messages);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function handleError(errorMsg) {
  const contentDiv = getOrCreateAssistantMessageContent();
  const errorDiv = document.createElement('div');
  errorDiv.className = 'error-message';
  errorDiv.textContent = `Error: ${errorMsg}`;
  contentDiv.appendChild(errorDiv);
  state.currentAssistantMessage = null;
}

function getOrCreateAssistantMessageContent() {
  if (!state.currentAssistantMessage) {
    const row = document.createElement('div');
    row.className = 'message assistant streaming';
    row.innerHTML = `<strong>Assistant</strong><div class="message-content"></div>`;
    chatWindow.appendChild(row);
    state.currentAssistantMessage = row;
  }
  return state.currentAssistantMessage.querySelector('.message-content');
}

function findIterationContainer(loop) {
  if (!state.currentAssistantMessage) return null;
  return state.currentAssistantMessage.querySelector(`[data-loop="${loop}"]`);
}

function reattachIterationToggleListeners() {
  const toggleButtons = chatWindow.querySelectorAll('.iteration-toggle');
  toggleButtons.forEach(button => {
    // Remove old listeners by cloning and replacing
    const newButton = button.cloneNode(true);
    const container = newButton.closest('.iteration-container');
    
    newButton.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      container.classList.toggle('expanded');
    });
    
    button.replaceWith(newButton);
  });
}

function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;

  addMessage('user', text);
  messageInput.value = '';

  try {
    if (!ws || !wsConnected) {
      initializeWebSocket();
      // Wait for connection
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('WebSocket connection timeout')), 5000);
        const checkConnection = setInterval(() => {
          if (wsConnected) {
            clearTimeout(timeout);
            clearInterval(checkConnection);
            resolve();
          }
        }, 100);
      });
    }
    
    sendMessageWebSocket(text);
  } catch (error) {
    addMessage('assistant', 'Server not available. Start the backend and reload the page.');
    console.error(error);
  }
});

if (settingsButton) {
  settingsButton.addEventListener('click', () => {
    // Save the current chat window HTML before navigating away
    localStorage.setItem('fyp_chat_html', chatWindow.innerHTML);
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

// Restore chat window HTML if it was saved before navigation
const savedChatHtml = localStorage.getItem('fyp_chat_html');
if (savedChatHtml) {
  chatWindow.innerHTML = savedChatHtml;
  localStorage.removeItem('fyp_chat_html'); // Clear it after restoring
  // Re-attach event listeners to iteration toggle buttons
  reattachIterationToggleListeners();
}

// Initialize WebSocket on load
initializeWebSocket();
