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
const mapContainer = document.querySelector('#mapContainer');
const mapSection = document.querySelector('#mapSection');
const progressSteps = [...document.querySelectorAll('#progressSteps li')];
const dismissKeyboard = document.querySelector('#dismissKeyboard');
const appMenu = document.querySelector('#appMenu');
const menuButton = document.querySelector('#menuButton');
const closeMenu = document.querySelector('#closeMenu');
const prefersMap = document.querySelector('#prefersMap');
const webBackend = document.querySelector('#webBackend');
const routerRelease = document.querySelector('#routerRelease');
let conversationId = null;
let activeTimer = null;
let mapAppHtml = null;
let activeMapCleanup = null;
let lastMapResult = null;
let mapRenderGeneration = 0;

const languageMessage = '現在は日本語の乗換案内に対応しています。\n「〇〇駅から〇〇駅」のように、出発地と目的地を日本語で入力してください。';
const constraintMessage = '現在のデモ版では、駅・路線・交通機関の除外や指定条件にはまだ対応していません。\n条件を外して、出発地と目的地を入力してください。';
const unsupportedConstraints = [
  /[^、,。\s]+?(?:駅)?(?:だけ)?(?:は|を)?(?:避け(?:て|たい|る)|通りたくない|通らない|通らず|除外(?:して|する)?|除いて|外して)/u,
  /[^、,。\s]+?(?:駅)?(?:を)?経由(?:(?:は|が)?嫌|しない|したくない|したくありません)/u,
  /(?:[\p{L}\p{N}・ー]+線|JR|地下鉄|東京メトロ|都営地下鉄|バス|電車|鉄道|列車|新幹線|飛行機|航空|フェリー|船|モノレール|路面電車|タクシー|徒歩)\s*で/u,
  /(?:[\p{L}\p{N}・ー]+線|JR|地下鉄|東京メトロ|都営地下鉄|バス|電車|鉄道|列車|新幹線|飛行機|航空|フェリー|船|モノレール|路面電車|タクシー|徒歩)(?:は|を)?(?:使わない|使いたくない|使わず|避け(?:て|たい)?|嫌|なし|以外|だけ|のみ|限定|指定)/u,
  /(?:[\p{L}\p{N}・ー]+線|JR|地下鉄|東京メトロ|都営地下鉄|バス|電車|鉄道|列車|新幹線|飛行機|航空|フェリー|船|モノレール|路面電車|タクシー)(?:を|で|が)?(?:使って|使いたい|利用して|利用したい|乗って|乗りたい|行って|行きたい|経由|優先|希望|指定|いい)/u,
  /[^、,。\s]+?(?:駅)?(?:は|が)(?:嫌|いや)/u,
  /[^、,。\s]+?(?:駅)?(?:なし|以外)(?:で)?/u,
];

function rejectionMessage(value) {
  const normalized = value.normalize('NFKC').trim().replace(/\s+/gu, ' ');
  const letters = normalized.match(/\p{L}/gu) || [];
  let hasJapanese = false;
  let hasLatin = false;
  for (const letter of letters) {
    if (letter === '\u30fc' || letter === '\uff70') continue;
    if (/^[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]$/u.test(letter)) {
      hasJapanese = true;
    } else if (/^\p{Script=Latin}$/u.test(letter)) {
      hasLatin = true;
    } else {
      return languageMessage;
    }
  }
  if (hasLatin && !hasJapanese) return languageMessage;
  if (unsupportedConstraints.some((pattern) => pattern.test(normalized))) return constraintMessage;
  return null;
}

function publicText(value, fallback) {
  if (typeof value !== 'string' || !value.trim()) return fallback;
  const trimmed = value.trim();
  if (/^(?:[\[{]|```json)/i.test(trimmed) || /<\/?(?:start_function_call|end_function_call|tool_call)>|"(?:structuredContent|jsonrpc)"\s*:/i.test(trimmed)) return fallback;
  return trimmed;
}

function setProgress(activeIndex, completed = false) {
  progressSteps.forEach((step, index) => {
    step.classList.toggle('done', completed || index < activeIndex);
    step.classList.toggle('active', !completed && index === activeIndex);
  });
}

function blurInput() {
  promptInput.blur();
}

function openMenu() {
  blurInput();
  if (!appMenu.open) appMenu.showModal();
}

try {
  const savedPreference = window.localStorage.getItem('tentetsu.prefersMap');
  if (savedPreference !== null) prefersMap.checked = savedPreference === 'true';
} catch (_) {
  // Storage can be unavailable in privacy mode; the default remains enabled.
}

menuButton.addEventListener('click', openMenu);
document.querySelectorAll('[data-open-menu]').forEach((button) => button.addEventListener('click', openMenu));
closeMenu.addEventListener('click', () => appMenu.close());
appMenu.addEventListener('click', (event) => {
  if (event.target === appMenu) appMenu.close();
});
prefersMap.addEventListener('change', () => {
  try {
    window.localStorage.setItem('tentetsu.prefersMap', String(prefersMap.checked));
  } catch (_) {
    // The setting still applies for the current page.
  }
  if (!prefersMap.checked) {
    teardownMap();
  } else if (lastMapResult) {
    renderRouteMap(lastMapResult);
  }
});
dismissKeyboard.addEventListener('click', blurInput);

async function checkHealth() {
  try {
    const response = await fetch('api/health', { cache: 'no-store' });
    if (!response.ok) throw new Error('health check failed');
    const health = await response.json();
    serviceState.className = 'service-state ready';
    serviceState.querySelector('span:last-child').textContent = 'サーバー利用可能';
    webBackend.textContent = health.inference_backend === 'server' ? 'Webサーバー' : 'サーバー構成';
    routerRelease.textContent = publicText(health.router_release || health.adapter, '配備構成')
      .slice(0, 36);
  } catch (_) {
    serviceState.className = 'service-state error';
    serviceState.querySelector('span:last-child').textContent = '接続できません';
  }
}

document.querySelectorAll('[data-example]').forEach((button) => {
  button.addEventListener('click', () => {
    promptInput.value = button.dataset.example;
    if (!window.matchMedia('(hover: none) and (pointer: coarse)').matches) promptInput.focus();
  });
});

// ---- MCP Apps host bridge for ui://transit/route-map -----------------------
// The map app is third-party HTML from the Transit MCP. It runs in a sandboxed
// iframe (allow-scripts only, opaque origin) and talks JSON-RPC over
// postMessage: ui/initialize -> ui/notifications/initialized -> we push
// ui/notifications/tool-result, then follow ui/notifications/size-changed.
function teardownMap({ forgetResult = false } = {}) {
  mapRenderGeneration += 1;
  if (activeMapCleanup) {
    activeMapCleanup();
    activeMapCleanup = null;
  }
  mapSection.hidden = true;
  mapContainer.replaceChildren();
  if (forgetResult) lastMapResult = null;
}

async function fetchMapAppHtml() {
  if (mapAppHtml) return mapAppHtml;
  const response = await fetch('api/ui/route-map', { cache: 'no-store' });
  if (!response.ok) throw new Error('map ui unavailable');
  mapAppHtml = await response.text();
  return mapAppHtml;
}

function routeEndpointName(endpoint) {
  if (!endpoint || typeof endpoint !== 'object') return '';
  const value = typeof endpoint.name === 'string'
    ? endpoint.name
    : typeof endpoint.label === 'string' ? endpoint.label : '';
  return value.normalize('NFKC').trim().replace(/\s+/gu, ' ');
}

function sameRouteEndpoint(left, right) {
  if (!left || !right || typeof left !== 'object' || typeof right !== 'object') return false;
  const leftId = typeof left.id === 'string' ? left.id.trim() : '';
  const rightId = typeof right.id === 'string' ? right.id.trim() : '';
  if (leftId && rightId) return leftId === rightId;
  const leftName = routeEndpointName(left);
  const rightName = routeEndpointName(right);
  return Boolean(leftName && rightName && leftName === rightName);
}

function routeSeconds(value) {
  if (value === null || value === undefined || value === '' || typeof value === 'boolean') {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function mapPointForRole(option, role) {
  const points = option && option.map && Array.isArray(option.map.points)
    ? option.map.points
    : [];
  return points.find((point) => point && point.role === role) || null;
}

// The MCP map UI owns rendering, but some map responses omit access/egress
// walks from journey.legs. Add those legs to a detached display copy only;
// the source result kept by this host remains untouched.
function routeMapDisplayResult(mapResult) {
  if (!mapResult || typeof mapResult !== 'object') return mapResult;
  let displayResult;
  try {
    displayResult = typeof structuredClone === 'function'
      ? structuredClone(mapResult)
      : JSON.parse(JSON.stringify(mapResult));
  } catch (_) {
    return mapResult;
  }
  const structured = displayResult.structuredContent;
  if (!structured || typeof structured !== 'object' || !Array.isArray(structured.options)) {
    return displayResult;
  }

  structured.options.forEach((option) => {
    const journey = option && option.journey;
    const legs = journey && Array.isArray(journey.legs) ? journey.legs : null;
    if (!legs || legs.length === 0) return;
    const first = legs[0];
    const last = legs[legs.length - 1];
    if (!first || !last || typeof first !== 'object' || typeof last !== 'object') return;

    const origin = mapPointForRole(option, 'origin') || structured.from;
    const destination = mapPointForRole(option, 'destination') || structured.to;
    const journeyDeparture = routeSeconds(journey.departureSecs);
    const firstDeparture = routeSeconds(first.departureSecs);
    const lastArrival = routeSeconds(last.arrivalSecs);
    const journeyArrival = routeSeconds(journey.arrivalSecs);

    if (origin && first.from && !sameRouteEndpoint(origin, first.from)
        && journeyDeparture !== null && firstDeparture !== null
        && firstDeparture > journeyDeparture) {
      legs.unshift({
        kind: 'walk',
        from: origin,
        to: first.from,
        departureSecs: journeyDeparture,
        arrivalSecs: firstDeparture,
        durationSecs: firstDeparture - journeyDeparture,
      });
    }
    if (destination && last.to && !sameRouteEndpoint(last.to, destination)
        && lastArrival !== null && journeyArrival !== null
        && journeyArrival > lastArrival) {
      legs.push({
        kind: 'walk',
        from: last.to,
        to: destination,
        departureSecs: lastArrival,
        arrivalSecs: journeyArrival,
        durationSecs: journeyArrival - lastArrival,
      });
    }
  });
  return displayResult;
}

async function renderRouteMap(mapResult) {
  teardownMap();
  const generation = mapRenderGeneration;
  let html;
  try {
    html = await fetchMapAppHtml();
  } catch (_) {
    return; // Text answer already covers the route; the map is progressive.
  }
  if (generation !== mapRenderGeneration || mapResult !== lastMapResult || !prefersMap.checked) {
    return;
  }
  const displayResult = routeMapDisplayResult(mapResult);
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
        params: displayResult,
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
  mapSection.hidden = !prefersMap.checked;
}

function beginLoading() {
  resultPanel.hidden = false;
  loading.hidden = false;
  choices.hidden = true;
  choices.replaceChildren();
  teardownMap({ forgetResult: true });
  errorBox.hidden = true;
  elapsed.textContent = '';
  submitButton.disabled = true;
  setProgress(1);
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
    const data = await response.json().catch(() => null);
    if (!response.ok || !data || !data.ok) {
      throw new Error(publicText(data && data.message, '検索に失敗しました。時間をおいて再度お試しください。'));
    }
    answer.textContent = publicText(data.answer, '経路情報を読み取れませんでした。もう一度お試しください。');
    elapsed.textContent = Number.isFinite(data.elapsed_ms) ? `${(data.elapsed_ms / 1000).toFixed(1)} sec` : '';
    setProgress(2);
    if (data.kind === 'map' && data.map_result && !data.map_result.isError) {
      lastMapResult = data.map_result;
      if (prefersMap.checked) renderRouteMap(lastMapResult);
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
        if (!window.matchMedia('(hover: none) and (pointer: coarse)').matches) promptInput.focus();
        window.scrollTo({ top: promptInput.getBoundingClientRect().top + window.scrollY - 90, behavior: 'smooth' });
      } else {
        conversationId = null;
      }
    }
    setProgress(2, true);
  } catch (error) {
    errorBox.textContent = publicText(error && error.message, '検索に失敗しました。時間をおいて再度お試しください。');
    errorBox.hidden = false;
    setProgress(1);
  } finally {
    endLoading();
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;
  blurInput();
  const rejection = rejectionMessage(prompt);
  if (rejection) {
    resultPanel.hidden = false;
    loading.hidden = true;
    choices.hidden = true;
    teardownMap({ forgetResult: true });
    answer.textContent = '';
    resultTitle.textContent = '入力を確認してください';
    errorBox.textContent = rejection;
    errorBox.hidden = false;
    elapsed.textContent = '';
    setProgress(0);
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    return;
  }
  answer.textContent = '';
  resultTitle.textContent = '検索結果';
  await requestQuery(conversationId ? { conversation_id: conversationId, prompt } : { prompt });
});

checkHealth();
