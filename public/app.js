const chatWindow = document.getElementById('chatWindow');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const settingsButton = document.getElementById('settingsButton');
const newChatBtn = document.getElementById('newChatBtn');
const backendStatus = document.getElementById('backendStatus');
const BACKEND_URL = 'http://localhost:5050';
const BACKEND_WS_URL = 'ws://localhost:5050';

// WebSocket connection state
let ws = null;
let wsConnected = false;

// Default settings - will be replaced by backend defaults
let defaultSettings = {
  model: 'gemma3:4b',
  systemPrompt: 'The explanation must be clear and beginner-friendly.',
  temperature: 0.7,
  includeRag: true,
  includeTextGrad: true,
  textGradModel: 'gemma3:4b',
  textGradLoops: 1,
  textGradLossPrompt: 'Evaluate this answer. It should be factual, clear, and directly answer the question.',
  apiKey: '',
  textGradApiKey: '',
};

// Fetch default settings from backend
async function fetchDefaultSettings() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/defaults`);
    if (response.ok) {
      defaultSettings = await response.json();
      console.log('Default settings loaded from backend:', defaultSettings);
    } else {
      console.warn('Failed to fetch default settings from backend, using fallback');
    }
  } catch (error) {
    console.error('Error fetching default settings:', error);
    // Use fallback defaults defined above
  }
}

function loadSavedSettings() {
  return window.SettingsStore.load(defaultSettings, defaultSettings.models || []);
}

function loadCurrentChat() {
  const stored = localStorage.getItem('fyp_current_chat');
  if (!stored) return [];
  try {
    return JSON.parse(stored);
  } catch (error) {
    console.error('Failed to parse current chat', error);
    return [];
  }
}

function saveCurrentChat(messages) {
  try {
    localStorage.setItem('fyp_current_chat', JSON.stringify(messages));
  } catch (error) {
    console.error('Failed to save current chat', error);
  }
}

function loadSavedConversations() {
  const stored = localStorage.getItem('fyp_saved_conversations');
  if (!stored) return [];
  try {
    return JSON.parse(stored);
  } catch (error) {
    console.error('Failed to parse saved conversations', error);
    return [];
  }
}

function saveSavedConversations(conversations) {
  try {
    localStorage.setItem('fyp_saved_conversations', JSON.stringify(conversations));
  } catch (error) {
    console.error('Failed to save conversations', error);
  }
}

function saveConversation(messages, settings) {
  const conversations = loadSavedConversations();
  
  // Filter out the welcome message - but keep all user and assistant messages
  const welcomeText = 'Welcome! This competitive-programming evaluation platform is styled like ChatGPT. Open the settings page to customize model or evaluation settings.';
  const filteredMessages = messages.filter(m => {
    // Keep all messages except the welcome message from assistant
    if (m.role === 'assistant' && m.text === welcomeText) {
      return false;
    }
    return true;
  });
  
  if (filteredMessages.length === 0) {
    return; // Don't save if only welcome message
  }
  
  // Get title from first user message
  const firstUserMessage = filteredMessages.find(m => m.role === 'user');
  const title = firstUserMessage?.text?.substring(0, 50) || 'Untitled Conversation';
  
  const conversation = {
    id: Date.now(),
    timestamp: new Date().toISOString(),
    title: title,
    messages: filteredMessages,
    settings: settings
  };
  conversations.unshift(conversation);
  saveSavedConversations(conversations);
  return conversation;
}

const state = {
  messages: (() => {
    const loaded = loadCurrentChat();
    // Only use loaded chat if it was explicitly loaded from Examples
    const wasLoadedFromExamples = sessionStorage.getItem('fyp_loaded_from_examples') === 'true';
    if (loaded.length > 0 && wasLoadedFromExamples) {
      sessionStorage.removeItem('fyp_loaded_from_examples');
      return loaded;
    }
    // Otherwise start fresh
    return [
      {
        role: 'assistant',
        text: 'Welcome! This competitive-programming evaluation platform is styled like ChatGPT. Open the settings page to customize model, benchmark, or evaluation settings.',
      },
    ];
  })(),
  settings: loadSavedSettings(),
  currentAssistantMessage: null,
  currentTextGradDetails: null,
  currentPrompt: null,
  conversationStart: null,
  collectedEvents: [], // Store all streaming events for the response
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
    
    if (message.role === 'assistant' && message.html) {
      // If we have saved HTML (from previous response), render it with full structure
      row.innerHTML = `<strong>Assistant</strong><div class="message-content">${message.html}</div>`;
    } else if (message.role === 'assistant') {
      // Fallback: render as plain text
      const formattedText = formatMessage(message.text || '');
      row.innerHTML = `<strong>Assistant</strong><div class="message-content">${formattedText}</div>`;
    } else {
      // User messages
      const formattedText = formatMessage(message.text || '');
      row.innerHTML = `<strong>You</strong><div>${formattedText}</div>`;
    }
    chatWindow.appendChild(row);
  });
  
  // Reattach event listeners to iteration toggles after rendering
  reattachIterationToggleListeners();
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addMessage(role, text) {
  state.messages.push({ role, text });
  saveCurrentChat(state.messages);
  renderMessages();
}

function saveSettings(settings) {
  return window.SettingsStore.save(
    settings,
    defaultSettings,
    defaultSettings.models || [],
    { requireApiKeys: true }
  );
}

function setBackendStatus(connected) {
  if (!backendStatus) return;
  backendStatus.textContent = connected ? 'Backend: connected' : 'Backend: disconnected';
  backendStatus.style.color = connected ? '#a5f3fc' : '#fca5a5';
}

async function checkBackendStatus() {
  if (!backendStatus) return;
  try {
    const response = await fetch(`${BACKEND_URL}/api/status`);
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
  if (ws) {
    console.log('WebSocket already exists, state:', ws.readyState);
    if (ws.readyState === WebSocket.OPEN) {
      wsConnected = true;
      return;
    } else if (ws.readyState === WebSocket.CONNECTING) {
      console.log('WebSocket is still connecting...');
      return;
    }
  }
  
  // Connect directly to Python backend
  const wsUrl = `${BACKEND_WS_URL}/ws/chat`;
  console.log('Initializing WebSocket connection to:', wsUrl);
  
  ws = new WebSocket(wsUrl);
  
  ws.onopen = () => {
    console.log('✓ WebSocket connected to Python backend');
    wsConnected = true;
  };
  
  ws.onmessage = (event) => {
    try {
      console.log('📨 Received WebSocket message:', event.data.substring(0, 100));
      const message = JSON.parse(event.data);
      handleStreamingEvent(message);
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error, event.data);
    }
  };
  
  ws.onerror = (error) => {
    console.error('✗ WebSocket error:', error);
    wsConnected = false;
  };
  
  ws.onclose = () => {
    console.log('✗ WebSocket closed');
    wsConnected = false;
    ws = null;
  };
}

function sendMessageWebSocket(message) {
  if (!ws) {
    throw new Error('WebSocket not initialized');
  }
  if (ws.readyState !== WebSocket.OPEN) {
    console.error('WebSocket state:', ws.readyState, '(0=CONNECTING, 1=OPEN, 2=CLOSING, 3=CLOSED)');
    throw new Error('WebSocket not in OPEN state');
  }
  console.log('📤 Sending message via WebSocket');
  ws.send(JSON.stringify({
    message,
    settings: state.settings,
  }));
}

function handleStreamingEvent(event) {
  const { type, data, loop, answer, original_prompt, updated, original, context, prompt } = event;
  
  // Collect all events
  state.collectedEvents.push(event);
  
  switch (type) {
    case 'start':
      console.log('Processing started');
      break;
      
    case 'formatted_prompt':
      appendFormattedPrompt(prompt);
      break;
      
    case 'rag_context':
      appendRAGContext(context);
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

function appendFormattedPrompt(prompt) {
  const contentDiv = getOrCreateAssistantMessageContent();
  
  const promptSection = document.createElement('div');
  promptSection.className = 'formatted-prompt-section';
  promptSection.innerHTML = `
    <div class="prompt-header">
      <span class="prompt-icon">💬</span>
      <strong>Formatted Prompt Sent to Model</strong>
    </div>
    <div class="prompt-content markdown-body"></div>
  `;
  
  const promptContentDiv = promptSection.querySelector('.prompt-content');
  promptContentDiv.innerHTML = marked.parse(prompt);
  
  contentDiv.insertBefore(promptSection, contentDiv.firstChild);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendRAGContext(context) {
  const contentDiv = getOrCreateAssistantMessageContent();
  
  const ragSection = document.createElement('div');
  ragSection.className = 'rag-context-section';
  ragSection.innerHTML = `
    <div class="rag-header">
      <span class="rag-icon">📚</span>
      <strong>Knowledge Base Context</strong>
    </div>
    <div class="rag-content markdown-body"></div>
  `;
  
  const ragContentDiv = ragSection.querySelector('.rag-content');
  ragContentDiv.innerHTML = marked.parse(context);
  
  contentDiv.insertBefore(ragSection, contentDiv.firstChild);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function appendIterationStart(loop, originalPrompt) {
  const contentDiv = getOrCreateAssistantMessageContent();
  
  const iterationContainer = document.createElement('div');
  iterationContainer.className = 'iteration-container expanded';
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
  
  // Wait for DOM to be fully updated before capturing HTML
  setTimeout(() => {
    // Capture the entire rendered response as HTML
    const fullResponseHTML = contentDiv.innerHTML;
    
    // Add the complete assistant message to state.messages
    state.messages.push({
      role: 'assistant',
      text: answer, // Save the final answer text
      html: fullResponseHTML, // Save the full HTML rendering
      events: state.collectedEvents // Save all streaming events
    });
    
    // Save immediately to localStorage
    saveCurrentChat(state.messages);
    
    // Save the conversation to savedConversations when complete
    if (state.messages.length > 1) {
      saveConversation(state.messages, state.settings);
    }
    
    state.currentAssistantMessage = null;
    state.collectedEvents = []; // Reset collected events for next response
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }, 100); // Small delay to ensure DOM is fully updated
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
    const container = button.closest('.iteration-container');
    
    if (container) {
      newButton.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        container.classList.toggle('expanded');
      });
      
      button.replaceWith(newButton);
    }
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
    // Ensure WebSocket is initialized
    initializeWebSocket();
    
    // Wait for connection to be established
    let attempts = 0;
    while ((!ws || ws.readyState !== WebSocket.OPEN) && attempts < 50) {
      console.log(`Waiting for WebSocket... (attempt ${attempts + 1}/50)`);
      await new Promise(resolve => setTimeout(resolve, 100));
      attempts++;
    }
    
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket connection failed after 5 seconds');
    }
    
    console.log('WebSocket ready, sending message...');
    sendMessageWebSocket(text);
  } catch (error) {
    console.error('Error sending message:', error);
    addMessage('assistant', `Error: ${error.message}. Make sure the Python backend is running on localhost:5050.`);
  }
});

messageInput.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter' || event.shiftKey || event.isComposing) return;

  event.preventDefault();
  chatForm.requestSubmit();
});

if (settingsButton) {
  settingsButton.addEventListener('click', () => {
    // Save the current chat window HTML before navigating away
    localStorage.setItem('fyp_chat_html', chatWindow.innerHTML);
    window.navigateWithTransition('settings.html');
  });
}

if (newChatBtn) {
  newChatBtn.addEventListener('click', () => {
    // Clear current chat and start new one
    sessionStorage.removeItem('fyp_loaded_from_examples'); // Clear the load flag
    localStorage.removeItem('fyp_current_chat'); // Clear current chat
    state.messages = [
      {
        role: 'assistant',
        text: 'Welcome! This competitive-programming evaluation platform is styled like ChatGPT. Open the settings page to customize model or evaluation settings.',
      },
    ];
    saveCurrentChat(state.messages);
    renderMessages();
  });
}

// Initialize app
async function initializeApp() {
  // Fetch default settings from backend first
  await fetchDefaultSettings();
  
  // Reload state settings after defaults are fetched
  state.settings = loadSavedSettings();
  
  renderMessages();
  checkBackendStatus();
  if (backendStatus) {
    setInterval(checkBackendStatus, 5000);
  }

  // Initialize WebSocket on load
  initializeWebSocket();
}

// Run initialization when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeApp);
} else {
  initializeApp();
}
