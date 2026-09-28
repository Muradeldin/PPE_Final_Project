const apiBase = window.location.protocol === 'file:'
  ? 'http://localhost:8000'
  : (window.APP_API_URL || `${window.location.protocol}//${window.location.hostname}:8000`);

const startButton = document.querySelector('#startButton');
const stopButton = document.querySelector('#stopButton');
const connectionState = document.querySelector('#connectionState');
const pipelineState = document.querySelector('#pipelineState');
const lastUpdate = document.querySelector('#lastUpdate');
const sourceValue = document.querySelector('#sourceValue');
const radarLabel = document.querySelector('#radarLabel');
document.querySelector('#apiLabel').textContent = `API: ${apiBase}`;

function setPipelineState(running, message = '') {
  pipelineState.textContent = running ? 'Running' : 'Offline';
  pipelineState.style.color = running ? '#167346' : '';
  radarLabel.textContent = running ? 'ACTIVE' : 'STANDBY';
  lastUpdate.textContent = message || (running ? 'Detection loop is active' : 'Ready to start');
  startButton.disabled = running;
  stopButton.disabled = !running;
}

async function request(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, options);
  if (!response.ok) throw new Error(`Backend returned ${response.status}`);
  return response.json();
}

async function refreshStatus() {
  try {
    const status = await request('/status');
    connectionState.classList.add('connected');
    connectionState.lastElementChild.textContent = 'Backend connected';
    sourceValue.textContent = status.source.split('/').pop() || 'CCTV test video';
    setPipelineState(status.running);
  } catch (error) {
    connectionState.classList.remove('connected');
    connectionState.lastElementChild.textContent = 'Backend offline';
    setPipelineState(false, 'Start the Pi backend to connect');
  }
}

startButton.addEventListener('click', async () => {
  startButton.disabled = true;
  lastUpdate.textContent = 'Loading model and starting...';
  try {
    const result = await request('/start', { method: 'POST' });
    setPipelineState(true, result.status);
  } catch (error) {
    setPipelineState(false, 'Could not start detection');
  }
});

stopButton.addEventListener('click', async () => {
  stopButton.disabled = true;
  try {
    const result = await request('/stop', { method: 'POST' });
    setPipelineState(false, result.status);
  } catch (error) {
    lastUpdate.textContent = 'Could not reach backend';
  }
});

refreshStatus();
setInterval(refreshStatus, 5000);
