import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.post('/api/chat', (req, res) => {
  const { message, settings } = req.body;
  const userMessage = (message || '').trim();
  const systemPrompt = settings?.systemPrompt || 'default system prompt';
  const model = settings?.model || 'gemma3:4b';

  const reply = userMessage
    ? `Demo reply from ${model}: I received your message and can honor the system prompt: "${systemPrompt}".`
    : 'Send a message from the input box to start the demo chat.';

  setTimeout(() => {
    res.json({ reply });
  }, 650);
});

app.post('/api/settings', (req, res) => {
  res.json({ status: 'ok' });
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
