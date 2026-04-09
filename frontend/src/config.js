// API base URL — override via VITE_API_URL env variable in production
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function startSession() {
  const res = await fetch(`${API_BASE_URL}/session/start`, { method: 'POST' });
  if (!res.ok) throw new Error(`Start session failed: ${res.status}`);
  return res.json(); // { session_id, message }
}

export async function sendChat(sessionId, userMessage) {
  const res = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, user_message: userMessage }),
  });
  if (!res.ok) throw new Error(`Chat failed: ${res.status}`);
  return res.json(); // { response, ended, session_id, pipeline_steps?, message_type? }
}
