import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// REST API endpoints (optional - kept for compatibility)
app.post('/api/chat', (req, res) => {
  res.json({ 
    message: 'Use WebSocket for streaming. Connect to ws://localhost:5050/ws/chat'
  });
});

app.post('/api/settings', (req, res) => {
  res.json({ status: 'ok' });
});

app.get('/api/status', async (req, res) => {
  res.json({ status: 'ok', message: 'Use Python backend at localhost:5050' });
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Frontend served at http://localhost:${port}`);
  console.log(`WebSocket and API: ws://localhost:5050/ws/chat and http://localhost:5050`);
});
