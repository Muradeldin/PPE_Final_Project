// Edge PPE Monitor: talks only to Supabase. The Pi syncs with the same tables (see app.py),
// so the website never needs to reach the Pi directly.
const db = supabase.createClient(window.PPE_CONFIG.supabaseUrl, window.PPE_CONFIG.supabaseKey);

const DEVICE_ID = 'pi';
const POLL_MS = 3000;
const ONLINE_SECONDS = 20;        // Pi counts as online if its last heartbeat is this recent
const PHOTO_LINK_SECONDS = 3600;  // Photos are private; signed links are valid this long

const $ = (selector) => document.querySelector(selector);
const connectionState = $('#connectionState');
const pipelineState = $('#pipelineState');
const lastUpdate = $('#lastUpdate');
const sourceValue = $('#sourceValue');
const modelNote = $('#modelNote');
const radarLabel = $('#radarLabel');
const startButton = $('#startButton');
const stopButton = $('#stopButton');

let device = null;
let violations = [];
let filter = 'open';
let seenIds = null;              // Violation ids already shown, so new ones can be highlighted
const photoLinks = new Map();    // image_path -> { url, expiresAt }
let pollTimer = null;

// ---- Helpers ------------------------------------------------------------------

function setConnection(online, text) {
  connectionState.classList.toggle('connected', online);
  connectionState.lastElementChild.textContent = text;
}

function timeAgo(date) {
  const seconds = Math.round((Date.now() - date) / 1000);
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return date.toLocaleDateString();
}

function isToday(date) {
  return date.toDateString() === new Date().toDateString();
}

function sourceName(source) {
  if (!source) return '—';
  if (/^\d+$/.test(source)) return `Camera ${source}`;
  return source.split(/[\\/]/).pop();
}

// ---- Sign in / out ------------------------------------------------------------

function showView(session) {
  const signedIn = Boolean(session);
  $('#loginView').hidden = signedIn;
  $('#appView').hidden = !signedIn;
  $('#account').hidden = !signedIn;

  if (signedIn) {
    $('#accountEmail').textContent = session.user.email;
    startPolling();
  } else {
    stopPolling();
    setConnection(false, 'Signed out');
  }
}

$('#loginForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = $('#signInButton');
  button.disabled = true;
  $('#loginError').textContent = '';

  const { error } = await db.auth.signInWithPassword({
    email: $('#email').value.trim(),
    password: $('#password').value,
  });
  if (error) $('#loginError').textContent = error.message;
  button.disabled = false;
});

$('#signOutButton').addEventListener('click', () => db.auth.signOut());

db.auth.onAuthStateChange((_event, session) => showView(session));

// ---- Pi status and Start/Stop -------------------------------------------------

function renderDevice() {
  const lastSeen = device?.last_seen ? new Date(device.last_seen) : null;
  const online = lastSeen !== null && (Date.now() - lastSeen) / 1000 < ONLINE_SECONDS;
  const desired = Boolean(device?.desired_running);
  const running = Boolean(device?.is_running) && online;

  setConnection(online, online ? 'Pi online' : 'Pi offline');

  let state, note;
  if (desired && running) {
    state = 'Running';
    note = 'Detection loop is active';
  } else if (desired) {
    state = 'Starting…';
    note = online ? 'Loading the model on the Pi' : 'Starts when the Pi connects';
  } else if (running) {
    state = 'Stopping…';
    note = 'Waiting for the Pi';
  } else {
    state = 'Offline';
    note = online ? 'Ready to start' : (lastSeen ? `Pi last seen ${timeAgo(lastSeen)}` : 'The Pi has not connected yet');
  }

  pipelineState.textContent = state;
  pipelineState.style.color = state === 'Running' ? '#167346' : '';
  lastUpdate.textContent = note;
  radarLabel.textContent = running ? 'ACTIVE' : (online ? 'STANDBY' : 'OFFLINE');
  sourceValue.textContent = sourceName(device?.source);
  modelNote.textContent = running && device.fps ? `320 px · ${device.fps.toFixed(1)} FPS on the Pi` : '320 px inference';

  startButton.disabled = desired;
  stopButton.disabled = !desired;

  updateLiveView(online, running);
}

// ---- Live view ----------------------------------------------------------------
// The Pi streams MJPEG through Tailscale Funnel; the link (with its secret token) comes from
// device_status, which only signed-in users can read. The stream only runs while it is on screen.

const liveImage = $('#liveImage');
const livePlaceholder = $('#livePlaceholder');
let liveFailedAt = 0;

function showLivePlaceholder(message) {
  if (liveImage.dataset.src) {
    liveImage.src = 'data:,';  // Closes the connection to the Pi
    delete liveImage.dataset.src;
  }
  liveImage.hidden = true;
  livePlaceholder.hidden = false;
  livePlaceholder.textContent = message;
}

function updateLiveView(online, running) {
  const url = device?.stream_url;
  if (!online) return showLivePlaceholder('The Pi is offline');
  if (!running) return showLivePlaceholder('Start detection to see the camera');
  if (!url) return showLivePlaceholder('The Pi has not reported a stream address');
  if (document.visibilityState !== 'visible') return showLivePlaceholder('Paused while the tab is hidden');
  if (Date.now() - liveFailedAt < 10_000) return;  // Recently failed: keep the error message, retry later

  if (liveImage.dataset.src !== url) {
    liveImage.dataset.src = url;
    liveImage.src = url;
  }
  liveImage.hidden = false;
  livePlaceholder.hidden = true;
}

liveImage.addEventListener('error', () => {
  if (!liveImage.dataset.src) return;  // Our own 'data:,' reset, not a real failure
  liveFailedAt = Date.now();
  showLivePlaceholder("Can't reach the Pi's live stream");
});

document.addEventListener('visibilitychange', () => {
  if (device) renderDevice();
});

async function refreshDevice() {
  const { data, error } = await db.from('device_status').select('*').eq('id', DEVICE_ID).maybeSingle();
  if (error) throw error;
  device = data;
  renderDevice();
}

async function setDesiredRunning(value) {
  startButton.disabled = true;
  stopButton.disabled = true;
  const { error } = await db.from('device_status').update({ desired_running: value }).eq('id', DEVICE_ID);
  if (error) lastUpdate.textContent = `Could not update: ${error.message}`;
  await refreshDevice().catch(() => {});
}

startButton.addEventListener('click', () => setDesiredRunning(true));
stopButton.addEventListener('click', () => setDesiredRunning(false));

// ---- Violations ---------------------------------------------------------------

async function ensurePhotoLinks(rows) {
  const now = Date.now();
  const missing = rows
    .map((v) => v.image_path)
    .filter((path) => !photoLinks.has(path) || photoLinks.get(path).expiresAt < now + 60_000);
  if (missing.length === 0) return;

  const { data, error } = await db.storage.from('violations').createSignedUrls(missing, PHOTO_LINK_SECONDS);
  if (error) throw error;
  for (const item of data) {
    if (item.signedUrl) photoLinks.set(item.path, { url: item.signedUrl, expiresAt: now + PHOTO_LINK_SECONDS * 1000 });
  }
}

function renderStats() {
  const today = violations.filter((v) => isToday(new Date(v.created_at)));
  $('#statToday').textContent = today.length;
  $('#statHelmet').textContent = today.filter((v) => v.reasons.includes('no_helmet')).length;
  $('#statVest').textContent = today.filter((v) => v.reasons.includes('no_vest')).length;
  const open = violations.filter((v) => !v.acknowledged).length;
  $('#statOpen').textContent = open;
  $('#ackAllButton').disabled = open === 0;
}

function violationCard(v, isNew) {
  const created = new Date(v.created_at);
  const card = document.createElement('article');
  card.className = `violation-card${v.acknowledged ? ' acked' : ''}${isNew ? ' new' : ''}`;

  const img = document.createElement('img');
  img.alt = `Worker ${v.worker_id}`;
  img.loading = 'lazy';
  img.src = photoLinks.get(v.image_path)?.url || '';
  img.addEventListener('click', () => {
    $('#viewer img').src = img.src;
    $('#viewer').showModal();
  });

  const body = document.createElement('div');
  body.className = 'violation-body';

  const top = document.createElement('div');
  top.className = 'violation-top';
  const worker = document.createElement('span');
  worker.className = 'violation-worker';
  worker.textContent = `Worker ${v.worker_id}`;
  const time = document.createElement('span');
  time.className = 'violation-time';
  time.textContent = `${created.toLocaleTimeString()} · ${timeAgo(created)}`;
  time.title = created.toLocaleString();
  top.append(worker, time);

  const badges = document.createElement('div');
  badges.className = 'badges';
  if (v.reasons.includes('no_helmet')) badges.insertAdjacentHTML('beforeend', '<span class="badge badge-helmet">No helmet</span>');
  if (v.reasons.includes('no_vest')) badges.insertAdjacentHTML('beforeend', '<span class="badge badge-vest">No vest</span>');

  body.append(top, badges);

  if (!v.acknowledged) {
    const button = document.createElement('button');
    button.className = 'button button-secondary';
    button.type = 'button';
    button.textContent = 'Acknowledge';
    button.addEventListener('click', async () => {
      button.disabled = true;
      await db.from('violations').update({ acknowledged: true }).eq('id', v.id);
      refreshViolations().catch(() => {});
    });
    body.append(button);
  }

  card.append(img, body);
  return card;
}

function renderViolations() {
  const shown = filter === 'open' ? violations.filter((v) => !v.acknowledged) : violations;
  const grid = $('#violationGrid');
  grid.replaceChildren();

  if (shown.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = filter === 'open' ? 'No violations waiting for review' : 'No violations recorded yet';
    grid.append(empty);
  } else {
    for (const v of shown) grid.append(violationCard(v, seenIds !== null && !seenIds.has(v.id)));
  }
  seenIds = new Set(violations.map((v) => v.id));
}

async function refreshViolations() {
  const { data, error } = await db
    .from('violations')
    .select('*')
    .order('created_at', { ascending: false })
    .limit(200);
  if (error) throw error;

  await ensurePhotoLinks(data);
  const changed = JSON.stringify(data) !== JSON.stringify(violations);
  violations = data;
  renderStats();
  if (changed || seenIds === null) renderViolations();
}

document.querySelectorAll('[data-filter]').forEach((tab) => {
  tab.addEventListener('click', () => {
    filter = tab.dataset.filter;
    document.querySelectorAll('[data-filter]').forEach((t) => t.classList.toggle('active', t === tab));
    renderViolations();
  });
});

$('#ackAllButton').addEventListener('click', async () => {
  $('#ackAllButton').disabled = true;
  await db.from('violations').update({ acknowledged: true }).eq('acknowledged', false);
  refreshViolations().catch(() => {});
});

$('#viewer').addEventListener('click', () => $('#viewer').close());

// ---- Polling ------------------------------------------------------------------

async function refresh() {
  try {
    await Promise.all([refreshDevice(), refreshViolations()]);
  } catch (error) {
    setConnection(false, "Can't reach Supabase");
    console.error(error);
  }
}

function startPolling() {
  if (pollTimer) return;
  refresh();
  pollTimer = setInterval(refresh, POLL_MS);
}

function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = null;
  showLivePlaceholder('Start detection to see the camera');
  device = null;
  violations = [];
  seenIds = null;
  photoLinks.clear();
}
