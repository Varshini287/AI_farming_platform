document.addEventListener('DOMContentLoaded', () => {
  const form  = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');
  const log   = document.getElementById('chat-log');

  if (!form || !input || !log) {
    console.warn('Chat elements not found in HTML.');
    return;
  }

  // helper to append a line to the chat log
  function appendMsg(who, text) {
    const line = document.createElement('div');
    line.className = who === 'user' ? 'chat-user' : 'chat-bot';
    line.textContent = (who === 'user' ? 'User: ' : 'Bot: ') + text;
    log.appendChild(line);
    // auto-scroll to bottom
    log.scrollTop = log.scrollHeight;
  }

  form.addEventListener('submit', async (evt) => {
    evt.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    // show user's message
    appendMsg('user', message);
    input.value = '';

    try {
      const resp = await fetch('/chat', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ message })
      });

      if (!resp.ok) {
        // try to extract server error text
        const errText = await resp.text().catch(() => resp.statusText);
        throw new Error(errText || `HTTP ${resp.status}`);
      }

      const { response } = await resp.json();
      appendMsg('bot', response);
    } catch (err) {
      console.error('Chat error:', err);
      appendMsg('bot', 'Network error, please try again.');
    }
  });
});

// static/js/chat.js

// Grab the DOM nodes once
const chatLog     = document.getElementById('chat-log');
const chatInput   = document.getElementById('chat-input');
const sendButton  = document.getElementById('chat-send-btn');

// This is the function your button is trying to call:
async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;                // nothing to do
  appendMessage('You', text);       // show user message in UI
  chatInput.value = '';             // clear input

  try {
    // send to your Flask backend
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: text })
    });
    const j = await resp.json();
    appendMessage('Bot', j.reply);
  } catch (err) {
    console.error(err);
    appendMessage('Bot', '😞 Oops, something went wrong.');
  }
}

// Utility to append one message to the chat-log
function appendMessage(who, text) {
  const p = document.createElement('p');
  p.classList.add('chat-line');
  p.innerHTML = `<strong>${who}:</strong> ${text}`;
  chatLog.appendChild(p);
  chatLog.scrollTop = chatLog.scrollHeight;
}

// wire up the button & enter key
sendButton.onclick = sendMessage;
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') sendMessage();
});
