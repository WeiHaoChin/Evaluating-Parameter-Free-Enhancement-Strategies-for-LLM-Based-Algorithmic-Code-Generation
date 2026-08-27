(function () {
  const STORAGE_KEY = 'fyp_chat_settings';
  const SETTING_KEYS = [
    'model',
    'systemPrompt',
    'temperature',
    'maxOutputTokens',
    'includeRag',
    'includeTextGrad',
    'textGradModel',
    'textGradLoops',
    'textGradInternalMaxOutputTokens',
    'textGradLossPrompt',
    'apiKey',
    'textGradApiKey',
  ];

  function withoutMetadata(settings) {
    const clean = {};
    for (const key of SETTING_KEYS) {
      if (Object.prototype.hasOwnProperty.call(settings || {}, key)) {
        clean[key] = settings[key];
      }
    }
    return clean;
  }

  function normalize(settings, defaults = {}, models = [], options = {}) {
    if (!settings || typeof settings !== 'object' || Array.isArray(settings)) {
      throw new Error('Settings must be a JSON object.');
    }

    const merged = {
      ...withoutMetadata(defaults),
      ...withoutMetadata(settings),
    };

    // Older exports may contain null for optional Pydantic API-key fields.
    merged.apiKey = merged.apiKey ?? '';
    merged.textGradApiKey = merged.textGradApiKey ?? '';

    for (const key of ['model', 'textGradModel']) {
      if (typeof merged[key] !== 'string' || !merged[key].trim()) {
        throw new Error(`${key} must contain a model name.`);
      }
      merged[key] = merged[key].trim();
    }

    const temperature = Number(merged.temperature);
    if (!Number.isFinite(temperature) || temperature < 0 || temperature > 1) {
      throw new Error('Temperature must be between 0 and 1.');
    }
    const loops = Number(merged.textGradLoops);
    if (!Number.isInteger(loops) || loops < 1 || loops > 5) {
      throw new Error('TextGrad loops must be an integer between 1 and 5.');
    }
    for (const key of ['maxOutputTokens', 'textGradInternalMaxOutputTokens']) {
      const value = Number(merged[key]);
      if (!Number.isInteger(value) || value < 1) {
        throw new Error(`${key} must be a positive integer.`);
      }
      merged[key] = value;
    }
    for (const key of ['includeRag', 'includeTextGrad']) {
      if (typeof merged[key] !== 'boolean') throw new Error(`${key} must be true or false.`);
    }
    for (const key of ['systemPrompt', 'textGradLossPrompt', 'apiKey', 'textGradApiKey']) {
      if (typeof merged[key] !== 'string') throw new Error(`${key} must be text.`);
    }

    merged.temperature = Math.round(temperature * 10) / 10;
    merged.textGradLoops = loops;
    merged.apiKey = merged.apiKey.trim();
    merged.textGradApiKey = merged.textGradApiKey.trim();

    if (options.requireApiKeys) {
      const providers = options.modelProviders || defaults.modelProviders || {};
      const requiresApiKey = model => providers[model] !== 'ollama_local' && model !== 'mock-chat:1.0';
      if (requiresApiKey(merged.model) && !merged.apiKey) {
        throw new Error('API key is required for the selected primary model.');
      }
      if (
        merged.includeTextGrad &&
        requiresApiKey(merged.textGradModel) &&
        !merged.textGradApiKey
      ) {
        throw new Error('API key is required for the selected TextGrad model.');
      }
    }

    return withoutMetadata(merged);
  }

  function load(defaults = {}, models = []) {
    const stored = sessionStorage.getItem(STORAGE_KEY);
    if (!stored) return normalize(defaults, defaults, models);
    try {
      return normalize(JSON.parse(stored), defaults, models);
    } catch (error) {
      console.error('Discarding invalid saved settings:', error);
      sessionStorage.removeItem(STORAGE_KEY);
      return normalize(defaults, defaults, models);
    }
  }

  function save(settings, defaults = {}, models = [], options = {}) {
    const normalized = normalize(settings, defaults, models, options);
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(normalized));
    return normalized;
  }

  window.SettingsStore = { load, normalize, save };
})();
