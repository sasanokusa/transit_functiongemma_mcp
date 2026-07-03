const form = document.querySelector('#searchForm');
const promptInput = document.querySelector('#prompt');
const submitButton = document.querySelector('#submitButton');
const resultPanel = document.querySelector('#resultPanel');
const loading = document.querySelector('#loading');
const loadingTime = document.querySelector('#loadingTime');
const answer = document.querySelector('#answer');
const errorBox = document.querySelector('#error');
const elapsed = document.querySelector('#elapsed');
const serviceState = document.querySelector('#serviceState');
const choices = document.querySelector('#choices');
const resultTitle = document.querySelector('#resultTitle');
const tracePanel = document.querySelector('#tracePanel');
const traceOutput = document.querySelector('#traceOutput');
const mapContainer = document.querySelector('#mapContainer');
let conversationId = null;
let activeTimer = null;
let mapAppHtml = null;
let activeMapCleanup = null;

async function checkHealth() {
  try {
    const response = await fetch('api/health', { cache: 'no-store' });
    if (!response.ok) throw new Error('health check failed');
    serviceState.className = 'service-state ready';
    serviceState.querySelector('span:last-child').textContent = '利用できます';
  } catch (_) {
    serviceState.className = 'service-state error';
    serviceState.querySelector('span:last-child').textContent = '接続できません';
  }
}

document.querySelectorAll('[data-example]').forEach((button) => {
  button.addEventListener('click', () => {
    promptInput.value = button.dataset.example;
    promptInput.focus();
  });
});

// ---- MCP Apps host bridge for ui://transit/route-map -----------------------
// The map app is third-party HTML from the Transit MCP. It runs in a sandboxed
// iframe (allow-scripts only, opaque origin) and talks JSON-RPC over
// postMessage: ui/initialize -> ui/notifications/initialized -> we push
// ui/notifications/tool-result, then follow ui/notifications/size-changed.
function teardownMap() {
  if (activeMapCleanup) {
    activeMapCleanup();
    activeMapCleanup = null;
  }
  mapContainer.hidden = true;
  mapContainer.replaceChildren();
}

async function fetchMapAppHtml() {
  if (mapAppHtml) return mapAppHtml;
  const response = await fetch('api/ui/route-map', { cache: 'no-store' });
  if (!response.ok) throw new Error('map ui unavailable');
  mapAppHtml = await response.text();
  return mapAppHtml;
}

async function renderRouteMap(mapResult) {
  teardownMap();
  let html;
  try {
    html = await fetchMapAppHtml();
  } catch (_) {
    return; // Text answer already covers the route; the map is progressive.
  }
  const frame = document.createElement('iframe');
  frame.className = 'map-frame';
  frame.setAttribute('sandbox', 'allow-scripts');
  frame.setAttribute('title', '経路マップ');
  frame.setAttribute('loading', 'lazy');
  frame.srcdoc = html;

  const post = (message) => {
    if (frame.contentWindow) frame.contentWindow.postMessage(message, '*');
  };
  const onMessage = (event) => {
    if (event.source !== frame.contentWindow) return;
    const message = event.data;
    if (!message || typeof message !== 'object' || message.jsonrpc !== '2.0') return;
    if (message.id !== undefined && message.method === 'ui/initialize') {
      post({
        jsonrpc: '2.0',
        id: message.id,
        result: {
          protocolVersion: '2025-06-18',
          hostInfo: { name: 'transit-functiongemma-web', version: '1.0.0' },
          hostCapabilities: {},
          hostContext: { theme: 'light', displayMode: 'inline' },
        },
      });
      return;
    }
    if (message.method === 'ui/notifications/initialized') {
      post({
        jsonrpc: '2.0',
        method: 'ui/notifications/tool-result',
        params: mapResult,
      });
      return;
    }
    if (message.method === 'ui/notifications/size-changed') {
      const height = message.params && Number(message.params.height);
      if (Number.isFinite(height) && height > 0) {
        frame.style.height = `${Math.min(Math.max(Math.ceil(height), 260), 1400)}px`;
      }
      return;
    }
    if (message.id !== undefined && message.method) {
      post({
        jsonrpc: '2.0',
        id: message.id,
        error: { code: -32601, message: 'method not supported by this host' },
      });
    }
  };
  window.addEventListener('message', onMessage);
  activeMapCleanup = () => window.removeEventListener('message', onMessage);
  mapContainer.append(frame);
  mapContainer.hidden = false;
}

function beginLoading() {
  resultPanel.hidden = false;
  loading.hidden = false;
  choices.hidden = true;
  choices.replaceChildren();
  teardownMap();
  errorBox.hidden = true;
  tracePanel.hidden = true;
  tracePanel.open = false;
  traceOutput.textContent = '';
  elapsed.textContent = '';
  submitButton.disabled = true;
  resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const started = Date.now();
  activeTimer = window.setInterval(() => {
    const seconds = Math.floor((Date.now() - started) / 1000);
    loadingTime.textContent = `${seconds}秒 — 駅と路線を照合中…`;
  }, 1000);
}

function endLoading() {
  window.clearInterval(activeTimer);
  activeTimer = null;
  loading.hidden = true;
  submitButton.disabled = false;
  loadingTime.textContent = '駅と路線を照合中…';
}

async function requestQuery(payload) {
  beginLoading();

  try {
    const response = await fetch('api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.message || '検索に失敗しました。');
    answer.textContent = data.answer;
    elapsed.textContent = `${(data.elapsed_ms / 1000).toFixed(1)} sec`;
    if (data.trace) {
      traceOutput.textContent = JSON.stringify(data.trace, null, 2);
      tracePanel.hidden = false;
    }
    if (data.kind === 'map' && data.map_result) {
      renderRouteMap(data.map_result);
    }
    if (data.kind === 'selection') {
      conversationId = data.conversation_id;
      const selectionLabel = data.selection_label || '駅';
      resultTitle.textContent = `${selectionLabel}を選択`;
      choices.setAttribute('aria-label', `${selectionLabel}候補`);
      const question = document.createElement('p');
      question.className = 'choice-question';
      question.textContent = data.question;
      choices.append(question);
      data.choices.forEach((choice) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'choice-button';
        const name = document.createElement('strong');
        name.textContent = `${choice.index + 1}. ${choice.name}`;
        const detail = document.createElement('span');
        const coordinates = Number.isFinite(choice.lat) && Number.isFinite(choice.lon)
          ? `${choice.lat.toFixed(5)}, ${choice.lon.toFixed(5)}`
          : null;
        detail.textContent = [choice.description, coordinates].filter(Boolean).join(' · ') || '物理駅候補';
        button.append(name, detail);
        button.addEventListener('click', () => {
          answer.textContent = `${choice.name}を選択しました。`;
          requestQuery({ conversation_id: conversationId, selection: choice.index });
        });
        choices.append(button);
      });
      choices.hidden = false;
    } else {
      choices.setAttribute('aria-label', '候補');
      resultTitle.textContent = data.kind === 'selected'
        ? '選択しました'
        : data.kind === 'awaiting_route' ? '続けて入力' : '検索結果';
      if (data.kind === 'selected' && data.selected) {
        conversationId = data.conversation_id;
        const selectedName = String(data.selected.name || '選択した駅');
        const stationName = data.selected.kind === 'station' && !selectedName.endsWith('駅')
          ? `${selectedName}駅`
          : selectedName;
        const question = document.createElement('p');
        question.className = 'choice-question';
        question.textContent = 'この駅をどちらに使いますか？';
        choices.append(question);
        [
          { label: '出発地にする', value: `${stationName}から`, role: 'origin' },
          { label: '目的地にする', value: `${stationName}まで`, role: 'destination' },
        ].forEach((action) => {
          const button = document.createElement('button');
          button.type = 'button';
          button.className = 'choice-button choice-action';
          const label = document.createElement('strong');
          label.textContent = action.label;
          const detail = document.createElement('span');
          detail.textContent = action.value;
          button.append(label, detail);
          button.addEventListener('click', async () => {
            await requestQuery({ conversation_id: conversationId, role: action.role });
          });
          choices.append(button);
        });
        choices.hidden = false;
      } else if (data.kind === 'awaiting_route') {
        conversationId = data.conversation_id;
        promptInput.value = '';
        promptInput.placeholder = data.placeholder || '不足している駅名を入力してください';
        promptInput.focus();
        window.scrollTo({ top: promptInput.getBoundingClientRect().top + window.scrollY - 90, behavior: 'smooth' });
      } else {
        conversationId = null;
      }
    }
  } catch (error) {
    errorBox.textContent = error.message || '検索に失敗しました。時間をおいて再度お試しください。';
    errorBox.hidden = false;
  } finally {
    endLoading();
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;
  answer.textContent = '';
  resultTitle.textContent = '検索結果';
  await requestQuery(conversationId ? { conversation_id: conversationId, prompt } : { prompt });
});

checkHealth();
