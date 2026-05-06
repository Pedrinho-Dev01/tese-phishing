// popup.js

const API_BASE = 'https://dpedrinho01-api-host.hf.space';

const EMOTION_META = {
  positive_arousal: { icon: '🎉', color: 'var(--emo-positive_arousal)' },
  warmth:           { icon: '🤗', color: 'var(--emo-warmth)'           },
  threat:           { icon: '😨', color: 'var(--emo-threat)'           },
  curiosity:        { icon: '🧐', color: 'var(--emo-curiosity)'        },
  confusion:        { icon: '😕', color: 'var(--emo-confusion)'        },
  sadness:          { icon: '😢', color: 'var(--emo-sadness)'          },
  relief:           { icon: '😮‍💨', color: 'var(--emo-relief)'           },
  neutral:          { icon: '😐', color: 'var(--emo-neutral)'          },
};

function extractEmailFromPage() {
  function detectClient() {
    const host = location.hostname;
    if (host === 'mail.google.com') return 'gmail';
    if (host.includes('outlook')) return 'outlook';
    return null;
  }

  function extractGmail() {
    const subjectEl = document.querySelector('h2[data-thread-perm-id], h2.hP');
    const subject = subjectEl ? subjectEl.innerText.trim() : '';

    const msgContainers = document.querySelectorAll('.adn.ads');
    if (!msgContainers.length) {
      const alt = document.querySelectorAll('[data-message-id]');
      if (!alt.length) return null;
    }

    const expanded = [...document.querySelectorAll('.adn.ads')]
      .filter(el => !el.classList.contains('kQ') && !el.classList.contains('kx'));
    const container = expanded[expanded.length - 1] || document.querySelector('.adn.ads');

    if (!container) return null;

    const fromEl = container.querySelector('.gD');
    const from = fromEl ? (fromEl.getAttribute('email') || fromEl.innerText.trim()) : '';

    const bodyEl = container.querySelector('.a3s.aiL') ||
      container.querySelector('.a3s') ||
      container.querySelector('.ii.gt div');
    const body = bodyEl ? bodyEl.innerText.trim() : '';

    if (!body) return null;

    const parts = [];
    if (subject) parts.push(`Subject: ${subject}`);
    if (from) parts.push(`From: ${from}`);
    parts.push(body);
    return parts.join('\n');
  }

  function extractOutlook() {
    const subjectEl =
      document.querySelector('[data-testid="subject"]') ||
      document.querySelector('.allowTextSelection span') ||
      document.querySelector('h1[role="heading"]');
    const subject = subjectEl ? subjectEl.innerText.trim() : '';

    const fromEl =
      document.querySelector('[aria-label^="From"]') ||
      document.querySelector('.Fw\\(600\\)') ||
      document.querySelector('[data-testid="senderName"]');
    const from = fromEl ? fromEl.innerText.trim() : '';

    const bodyEl =
      document.querySelector('[data-testid="message-body"]') ||
      document.querySelector('.ReadMsgBody') ||
      document.querySelector('[aria-label="Message body"]') ||
      document.querySelector('.allowTextSelection');
    const body = bodyEl ? bodyEl.innerText.trim() : '';

    if (!body) return null;

    const parts = [];
    if (subject) parts.push(`Subject: ${subject}`);
    if (from) parts.push(`From: ${from}`);
    parts.push(body);
    return parts.join('\n');
  }

  const client = detectClient();
  if (client === 'gmail') return { text: extractGmail(), client };
  if (client === 'outlook') return { text: extractOutlook(), client };
  return { text: null, client: null };
}

async function analyseText(text) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model: 'ensemble' }),
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  const spam = await res.json();

  const res2 = await fetch(`${API_BASE}/predict/emotion`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res2.ok) throw new Error(`Emotion API error ${res2.status}`);
  const emotion = await res2.json();

  return { spam, emotion };
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function setLoading(on) {
  const btn = document.getElementById('scan-btn');
  btn.disabled = on;
  document.getElementById('spinner').style.display = on ? 'block' : 'none';
  document.getElementById('btn-label').textContent  = on ? 'Scanning…' : 'Scan email';
}

function showError(msg) {
  const el = document.getElementById('error-banner');
  el.textContent = '⚠ ' + msg;
  el.style.display = 'block';
}

function hideError() {
  document.getElementById('error-banner').style.display = 'none';
}

function hideResults() {
  document.getElementById('result-card').classList.remove('visible');
  document.getElementById('emotion-card').classList.remove('visible');
  document.getElementById('prob-bar').style.width = '0%';
}

function hideIdle() {
  document.getElementById('idle-state').style.display = 'none';
}

function showIdle() {
  document.getElementById('idle-state').style.display = 'flex';
}

function extractTextFromInjectionResult(result) {
  if (!result || typeof result !== 'object') return { text: null, client: null };
  return {
    text: result.text ?? null,
    client: result.client ?? null,
  };
}

// ── Spam rendering ────────────────────────────────────────────────────────────

function modelColorClass(prob, threshold) {
  if (prob >= 0.5)        return 'col-spam';
  if (prob >= threshold)  return 'col-maybe';
  return 'col-ham';
}

function modelVerdictLabel(prob, threshold) {
  if (prob >= 0.5)        return '🚨 Spam';
  if (prob >= threshold)  return '⚠️ Maybe';
  return '✅ Ham';
}

function renderSpam(data) {
  const pct    = Math.round(data.spam_probability * 100);
  const card   = document.getElementById('result-card');
  const isSpam  = data.is_spam;
  const isMaybe = data.maybe_spam;

  card.className = 'result-card ' + (isSpam ? 'spam' : isMaybe ? 'maybe' : 'ham');

  document.getElementById('verdict-icon').textContent =
    isSpam ? '🚨' : isMaybe ? '⚠️' : '✅';
  document.getElementById('verdict-text').textContent =
    isSpam ? 'Spam detected' : isMaybe ? 'Maybe spam' : 'Looks clean';
  document.getElementById('verdict-sub').textContent =
    isSpam  ? 'Classified as spam by the ensemble' :
    isMaybe ? 'Some spam signals — review carefully' :
              'No spam signals found';

  document.getElementById('prob-big').textContent = pct + '%';
  document.getElementById('threshold-label').textContent =
    `Threshold: ${Math.round(data.ensemble_threshold * 100)}%`;

  requestAnimationFrame(() => {
    document.getElementById('prob-bar').style.width = pct + '%';
  });

  if (data.roberta) {
    const rp = Math.round(data.roberta.spam_probability * 100);
    const rc = modelColorClass(data.roberta.spam_probability, data.roberta.threshold);
    document.getElementById('roberta-prob').textContent = rp + '%';
    document.getElementById('roberta-prob').className   = 'm-prob ' + rc;
    document.getElementById('roberta-verdict').textContent = modelVerdictLabel(data.roberta.spam_probability, data.roberta.threshold);
    document.getElementById('roberta-verdict').className   = 'm-verdict ' + rc;
  }
  if (data.electra) {
    const ep = Math.round(data.electra.spam_probability * 100);
    const ec = modelColorClass(data.electra.spam_probability, data.electra.threshold);
    document.getElementById('electra-prob').textContent = ep + '%';
    document.getElementById('electra-prob').className   = 'm-prob ' + ec;
    document.getElementById('electra-verdict').textContent = modelVerdictLabel(data.electra.spam_probability, data.electra.threshold);
    document.getElementById('electra-verdict').className   = 'm-verdict ' + ec;
  }

  requestAnimationFrame(() => card.classList.add('visible'));
}

// ── Emotion rendering ─────────────────────────────────────────────────────────

function renderEmotion(data) {
  const card     = document.getElementById('emotion-card');
  const detected = data.detected_emotions || [];
  const scores   = data.all_scores || [];

  // Subtitle
  document.getElementById('emotion-subtitle').textContent =
    detected.length === 0
      ? 'No strong emotions detected.'
      : `${detected.length} emotion${detected.length > 1 ? 's' : ''} detected.`;

  // Detected chips
  const chipsWrap = document.getElementById('emotion-detected-wrap');
  chipsWrap.innerHTML = '';
  detected.forEach((emo, i) => {
    const meta = EMOTION_META[emo] || { icon: '•', color: 'var(--muted)' };
    const chip = document.createElement('span');
    chip.className = 'emo-chip';
    chip.style.cssText =
      `color:${meta.color};border-color:${meta.color}33;background:${meta.color}18;animation-delay:${i * 50}ms`;
    chip.innerHTML =
      `<span class="emo-chip-dot"></span>${meta.icon} ${emo.replace('_', ' ')}`;
    chipsWrap.appendChild(chip);
  });

  // Bar chart
  const barsEl = document.getElementById('emotion-bars');
  barsEl.innerHTML = '';
  scores.forEach((score, i) => {
    const meta  = EMOTION_META[score.emotion] || { icon: '•', color: 'var(--muted)' };
    const pct   = Math.round(score.probability * 100);
    const label = score.emotion.replace('_', ' ');
    const row   = document.createElement('div');
    row.className = 'emo-bar-row';
    row.style.animationDelay = `${i * 35}ms`;
    row.innerHTML = `
      <div class="emo-bar-label">
        <span class="emo-bar-label-icon">${meta.icon}</span>
        <span>${label}</span>
      </div>
      <div class="emo-bar-track">
        <div class="emo-bar-fill" data-pct="${pct}" style="background:${meta.color}"></div>
      </div>
      <div class="emo-bar-pct" style="color:${score.detected ? meta.color : 'var(--muted)'}">${pct}%</div>
    `;
    barsEl.appendChild(row);
  });

  requestAnimationFrame(() => {
    barsEl.querySelectorAll('.emo-bar-fill').forEach(fill => {
      fill.style.width = fill.dataset.pct + '%';
    });
  });

  requestAnimationFrame(() => card.classList.add('visible'));
}

// ── Main scan logic ───────────────────────────────────────────────────────────

async function runScan() {
  hideError();
  hideResults();
  setLoading(true);

  try {
    // 1. Get the active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    // 2. Extract the open email directly from the active page.
    const [injectionResult] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: extractEmailFromPage,
    });
    const extracted = extractTextFromInjectionResult(injectionResult?.result);

    if (!extracted || !extracted.text) {
      showIdle();
      showError(
        extracted?.client
          ? 'Could not find an open email. Click on an email to open it, then scan again.'
          : 'No supported email client detected. Open Gmail or Outlook in this tab.'
      );
      setLoading(false);
      return;
    }

    hideIdle();

    // 3. Call the API directly from the popup.
    const result = await analyseText(extracted.text);

    renderSpam(result.spam);
    renderEmotion(result.emotion);

  } catch (err) {
    showError(
      err.message.includes('Cannot access') || err.message.includes('connect')
        ? 'Cannot reach the API. Check your internet connection and try again.'
        : err.message
    );
    showIdle();
  } finally {
    setLoading(false);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('scan-btn').addEventListener('click', runScan);
});
