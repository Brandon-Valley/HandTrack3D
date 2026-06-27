'use strict';

const VERSION = '2026-06-27-natural-model-only';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const FACE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task';
const HAND_ASSET_URL = 'https://cdn.jsdelivr.net/gh/GodotVR/godot_openxr_for_godot_3.x@e983d79365cf5a59052e6242dba083a467020dee/demo/addons/godot-openxr/assets/valve_hand_models/right_hand.glb';
const BODY_ASSET_URL = 'https://threejs.org/examples/models/gltf/Xbot.glb';
const FACE_ASSET_URL = 'https://threejs.org/examples/models/gltf/RobotExpressive/RobotExpressive.glb';

const HAND_CONN = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],[5,17]];
const BODY_CONN = [[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[27,31],[29,31],[24,26],[26,28],[28,30],[28,32],[30,32],[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[0,11],[0,12]];
const UPPER_CONN = [[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[0,11],[0,12]];
const FACE_IDS = [10,152,234,454,1,4,33,133,263,362,61,291,13,14,0,17,50,280,199,175,151,9];
const FACE_CONN = [[0,2],[0,3],[2,1],[3,1],[6,7],[8,9],[10,12],[12,11],[11,13],[13,10],[4,5],[14,15],[20,21],[18,19]];
const TIP = new Set([4,8,12,16,20]);

let THREE, GLTFLoader, FilesetResolver, HandLandmarker, FaceLandmarker, PoseLandmarker, filesetPromise;
let renderer, scene, camera3d, root, grid, tracker;
let mode = 'hand';
let modelStyle = 'natural';
let trackerMode = 'waiting';
let running = false;
let loadingTracker = false;
let currentConnections = HAND_CONN;
let points = [];
let target = [];
let smooth = [];
let lines = [];
let joints = [];
let extras = [];
let assetRoot = null;
let assetLoadToken = 0;
let assetStatus = 'none';
let analyzed = 0;
let tracked = 0;
let misses = 0;
let triedRelaxed = false;
let lastDetect = 0;
let lastFrame = performance.now();
let fpsValue = 0;
let yAxis, v1, v2, v3;
let lastAssetCenter = null;

const el = {
  app: document.getElementById('app'),
  stage: document.getElementById('stage'),
  fallback: document.getElementById('stageFallback'),
  video: document.getElementById('video'),
  overlay: document.getElementById('overlay'),
  startPanel: document.getElementById('startPanel'),
  startButton: document.getElementById('startButton'),
  startLog: document.getElementById('startLog'),
  startError: document.getElementById('startError'),
  statusCard: document.getElementById('statusCard'),
  statusText: document.getElementById('statusText'),
  stateDot: document.getElementById('stateDot'),
  handedness: document.getElementById('handedness'),
  confidence: document.getElementById('confidence'),
  fps: document.getElementById('fps'),
  tracked: document.getElementById('trackedFrames'),
  analyzed: document.getElementById('analyzedFrames'),
  tracker: document.getElementById('trackerMode'),
  videoSize: document.getElementById('videoSize'),
  model: document.getElementById('modelMode'),
  noHand: document.getElementById('noHandBanner')
};
let overlayCtx = el.overlay.getContext('2d', { alpha: true });

setupPanels();
resizeOverlay();
showLog('Ready. Version: ' + VERSION);
updateLabels();
el.startButton.addEventListener('click', startDemo);
window.addEventListener('resize', resizeAll);
window.addEventListener('orientationchange', () => setTimeout(resizeAll, 250));

function setupPanels() {
  const css = document.createElement('style');
  css.textContent = `
    #statusCard{transition:all .16s ease}#statusToggle{appearance:none;width:100%;min-height:0;margin:9px 0 0;padding:7px 9px;border-radius:999px;border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.08);color:var(--text);font-size:11px;font-weight:950;box-shadow:none}
    #statusCard.collapsed{width:auto;min-width:0;max-width:calc(100vw - 88px);padding:8px 9px;border-radius:999px;display:flex;align-items:center;gap:8px}#statusCard.collapsed #title{margin:0;font-size:13px;line-height:1;white-space:nowrap}#statusCard.collapsed .statusLine{font-size:0;gap:0}#statusCard.collapsed #statusText,#statusCard.collapsed #metrics{display:none}#statusCard.collapsed #statusToggle{width:auto;margin:0;padding:6px 8px;font-size:11px;background:rgba(255,255,255,.1)}
    #modePanel{position:absolute;z-index:8;top:max(12px,env(safe-area-inset-top));left:50%;transform:translateX(-50%);width:min(390px,calc(100vw - 112px));pointer-events:auto}#modeToggle{width:100%;min-height:42px;padding:10px 13px;border:1px solid rgba(255,255,255,.22);border-radius:999px;background:rgba(8,12,22,.9);color:#fff;box-shadow:0 18px 54px rgba(0,0,0,.36);backdrop-filter:blur(18px);font-size:13px}#modeSheet{display:none;margin-top:8px;padding:10px;border:1px solid rgba(255,255,255,.18);border-radius:18px;background:rgba(8,12,22,.96);box-shadow:0 18px 54px rgba(0,0,0,.42);backdrop-filter:blur(18px)}#modePanel.open #modeSheet{display:block}.controlLabel{margin:0 0 7px;color:var(--muted);font-size:11px;font-weight:900;text-transform:uppercase;letter-spacing:.08em}.buttonRow{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px}.buttonRow.two{grid-template-columns:repeat(2,1fr);margin-bottom:0}.modeChoice{min-height:38px;padding:9px 7px;border-radius:999px;border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.07);color:var(--text);box-shadow:none;font-size:12px}.modeChoice.active{background:linear-gradient(180deg,#e4fbff,#65d9ff);color:#03131d;border-color:transparent}.uiOpen #tips,.uiOpen #noHandBanner{opacity:0;pointer-events:none}
    @media(max-width:560px){#statusCard.collapsed{max-width:calc(100vw - 84px)}#modePanel{top:calc(max(12px,env(safe-area-inset-top)) + 52px);left:max(12px,env(safe-area-inset-left));right:auto;transform:none;width:min(305px,calc(100vw - 96px))}#modeToggle{min-height:38px;font-size:12px;padding:9px 11px}.buttonRow{grid-template-columns:repeat(2,1fr)}}
  `;
  document.head.appendChild(css);

  const statsButton = document.createElement('button');
  statsButton.id = 'statusToggle';
  statsButton.type = 'button';
  statsButton.textContent = 'Stats';
  el.statusCard.insertBefore(statsButton, document.getElementById('metrics'));
  el.statusCard.classList.add('collapsed');
  statsButton.addEventListener('click', () => {
    closeModeMenu();
    const closed = el.statusCard.classList.toggle('collapsed');
    statsButton.textContent = closed ? 'Stats' : 'Hide stats';
  });

  const panel = document.createElement('div');
  panel.id = 'modePanel';
  panel.innerHTML = `
    <button id='modeToggle' type='button'>Hand / Natural</button>
    <div id='modeSheet' class='card'>
      <p class='controlLabel'>Tracking mode</p>
      <div class='buttonRow'><button class='modeChoice' data-mode='hand'>Hand</button><button class='modeChoice' data-mode='face'>Face</button><button class='modeChoice' data-mode='upper'>Upper</button><button class='modeChoice' data-mode='body'>Body</button></div>
      <p class='controlLabel'>Model style</p>
      <div class='buttonRow two'><button class='modeChoice' data-style='stick'>Stick</button><button class='modeChoice' data-style='natural'>Natural rig</button></div>
    </div>`;
  el.app.appendChild(panel);
  panel.querySelector('#modeToggle').addEventListener('click', () => {
    el.statusCard.classList.add('collapsed');
    statsButton.textContent = 'Stats';
    panel.classList.toggle('open');
    document.body.classList.toggle('uiOpen', panel.classList.contains('open'));
  });
  panel.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => switchMode(button.dataset.mode)));
  panel.querySelectorAll('[data-style]').forEach(button => button.addEventListener('click', () => switchStyle(button.dataset.style)));
  refreshButtons();
}

function closeModeMenu() {
  const panel = document.getElementById('modePanel');
  if (panel) panel.classList.remove('open');
  document.body.classList.remove('uiOpen');
}

function refreshButtons() {
  document.querySelectorAll('[data-mode]').forEach(button => button.classList.toggle('active', button.dataset.mode === mode));
  document.querySelectorAll('[data-style]').forEach(button => button.classList.toggle('active', button.dataset.style === modelStyle));
  const toggle = document.getElementById('modeToggle');
  if (toggle) toggle.textContent = label(mode) + ' / ' + label(modelStyle);
}

function label(value) {
  if (value === 'hand') return 'Hand';
  if (value === 'face') return 'Face';
  if (value === 'upper') return 'Upper body';
  if (value === 'body') return 'Full body';
  if (value === 'stick') return 'Stick';
  return 'Natural rig';
}

function updateLabels() {
  el.model.textContent = 'Model: ' + label(mode) + ' ' + label(modelStyle) + (assetStatus && assetStatus !== 'none' ? ' / asset ' + assetStatus : '');
  el.tracker.textContent = 'Tracker: ' + trackerMode;
}

async function switchMode(nextMode) {
  mode = nextMode;
  analyzed = 0;
  tracked = 0;
  misses = 0;
  triedRelaxed = false;
  closeModeMenu();
  rebuildModel();
  refreshButtons();
  updateLabels();
  clearOverlay();
  if (running) await initTracker(false);
}

function switchStyle(nextStyle) {
  modelStyle = nextStyle;
  closeModeMenu();
  rebuildModel();
  refreshButtons();
  updateLabels();
}

async function startDemo() {
  el.startButton.disabled = true;
  el.startButton.textContent = 'Starting camera...';
  el.startError.style.display = 'none';
  setStatus('Requesting camera...', 'warn');
  try {
    await startCamera();
    el.startButton.textContent = 'Loading libraries...';
    setStatus('Loading libraries...', 'warn');
    await loadLibraries();
    initScene();
    rebuildModel();
    el.startButton.textContent = 'Loading tracker...';
    await initTracker(false);
    running = true;
    el.startPanel.style.display = 'none';
    setStatus('Tracking is running. Use the mode pill to switch.', 'warn');
  } catch (error) {
    console.error(error);
    stopCamera();
    running = false;
    el.startButton.disabled = false;
    el.startButton.textContent = 'Try again';
    el.startError.textContent = friendlyError(error);
    el.startError.style.display = 'block';
    setStatus('Could not start.', 'bad');
  }
}

async function startCamera() {
  if (!window.isSecureContext) throw new Error('Camera access needs HTTPS or localhost.');
  const attempts = [
    { audio: false, video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 60 } } },
    { audio: false, video: { facingMode: 'user' } },
    { audio: false, video: true }
  ];
  let lastError = null;
  for (const options of attempts) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(options);
      el.video.srcObject = stream;
      el.video.setAttribute('playsinline', '');
      el.video.muted = true;
      await waitForVideo(el.video);
      await el.video.play();
      resizeOverlay();
      return;
    } catch (error) {
      lastError = error;
      stopCamera();
    }
  }
  throw lastError || new Error('No camera stream found.');
}

function waitForVideo(video) {
  return new Promise((resolve, reject) => {
    if (video.videoWidth > 0 && video.videoHeight > 0 && video.readyState >= 2) {
      resolve();
      return;
    }
    const timer = setTimeout(() => reject(new Error('Camera opened, but video did not become ready.')), 9000);
    function ready() {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        clearTimeout(timer);
        video.removeEventListener('loadedmetadata', ready);
        video.removeEventListener('canplay', ready);
        resolve();
      }
    }
    video.addEventListener('loadedmetadata', ready);
    video.addEventListener('canplay', ready);
    video.play().catch(() => {});
  });
}

function stopCamera() {
  if (el.video.srcObject && el.video.srcObject.getTracks) el.video.srcObject.getTracks().forEach(track => track.stop());
  el.video.srcObject = null;
}

async function loadLibraries() {
  const modules = await Promise.all([
    import('three'),
    import('https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/loaders/GLTFLoader.js'),
    import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs')
  ]);
  THREE = modules[0];
  GLTFLoader = modules[1].GLTFLoader;
  FilesetResolver = modules[2].FilesetResolver;
  HandLandmarker = modules[2].HandLandmarker;
  FaceLandmarker = modules[2].FaceLandmarker;
  PoseLandmarker = modules[2].PoseLandmarker;
}

async function initTracker(relaxed) {
  loadingTracker = true;
  setStatus('Loading ' + label(mode).toLowerCase() + ' tracker...', 'warn');
  const fileset = await (filesetPromise || (filesetPromise = FilesetResolver.forVisionTasks(WASM_URL)));
  try { if (tracker && tracker.close) tracker.close(); } catch (error) {}
  const preferCpu = relaxed || mode === 'face';
  const delegate = preferCpu ? 'CPU' : 'GPU';
  try {
    if (mode === 'hand') {
      tracker = await HandLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate }, runningMode: 'VIDEO', numHands: 1, minHandDetectionConfidence: relaxed ? 0.10 : 0.22, minHandPresenceConfidence: relaxed ? 0.10 : 0.22, minTrackingConfidence: relaxed ? 0.10 : 0.22 });
    } else if (mode === 'face') {
      tracker = await FaceLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate }, runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true, minFaceDetectionConfidence: 0.12, minFacePresenceConfidence: 0.12, minTrackingConfidence: 0.12 });
    } else {
      tracker = await PoseLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate }, runningMode: 'VIDEO', numPoses: 1, minPoseDetectionConfidence: relaxed ? 0.10 : 0.20, minPosePresenceConfidence: relaxed ? 0.10 : 0.20, minTrackingConfidence: relaxed ? 0.10 : 0.20 });
    }
    trackerMode = label(mode) + ' ' + delegate;
  } catch (error) {
    if (!preferCpu) {
      loadingTracker = false;
      return initTracker(true);
    }
    throw error;
  } finally {
    loadingTracker = false;
    updateLabels();
  }
}

function initScene() {
  if (scene) return;
  yAxis = new THREE.Vector3(0, 1, 0);
  v1 = new THREE.Vector3();
  v2 = new THREE.Vector3();
  v3 = new THREE.Vector3();
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.localClippingEnabled = true;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  el.fallback.style.display = 'none';
  el.stage.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  camera3d = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 100);
  camera3d.position.set(0, 0.05, 8.3);
  scene.add(new THREE.HemisphereLight(0xeaf7ff, 0x15182c, 3));
  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(2.8, 4.4, 4.8);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x76e3ff, 1.5);
  rim.position.set(-4.4, 1, -2.4);
  scene.add(rim);
  grid = new THREE.Group();
  scene.add(grid);
  makeGrid();
  root = new THREE.Group();
  scene.add(root);
  requestAnimationFrame(animate);
}

function rebuildModel() {
  if (!root || !THREE) return;
  while (root.children.length) root.remove(root.children[0]);
  lines = [];
  joints = [];
  extras = [];
  assetRoot = null;
  lastAssetCenter = null;
  assetStatus = 'none';
  assetLoadToken++;
  currentConnections = mode === 'hand' ? HAND_CONN : mode === 'face' ? FACE_CONN : mode === 'upper' ? UPPER_CONN : BODY_CONN;
  const count = mode === 'hand' ? 21 : mode === 'face' ? FACE_IDS.length : 33;
  points = makeDefaultPoints(count);
  target = points.map(point => point.clone());
  smooth = points.map(point => point.clone());
  const lineMaterial = new THREE.MeshStandardMaterial({ color: modelStyle === 'stick' ? 0x90f4ff : 0xe9a47b, roughness: 0.5, transparent: true, opacity: modelStyle === 'stick' ? 0.95 : 0.42 });
  const jointMaterial = new THREE.MeshStandardMaterial({ color: modelStyle === 'stick' ? 0xffffff : 0xffd1b8, roughness: 0.45, transparent: true, opacity: modelStyle === 'stick' ? 1.0 : 0.46 });
  const cylinder = new THREE.CylinderGeometry(1, 1, 1, 18, 1, false);
  const sphere = new THREE.SphereGeometry(1, 20, 12);
  currentConnections.forEach(pair => {
    const mesh = new THREE.Mesh(cylinder, lineMaterial);
    mesh.userData = { a: pair[0], b: pair[1] };
    root.add(mesh);
    lines.push(mesh);
  });
  points.forEach((point, index) => {
    const mesh = new THREE.Mesh(sphere, jointMaterial);
    mesh.userData.index = index;
    root.add(mesh);
    joints.push(mesh);
  });
  if (modelStyle === 'natural') addNaturalExtras();
  if (modelStyle === 'natural') loadRiggedAssetInBackground();
  drawModel();
  updateLabels();
}

function makeDefaultPoints(count) {
  const out = [];
  for (let i = 0; i < count; i++) out.push(new THREE.Vector3((i % 6 - 2.5) * 0.18, (2 - Math.floor(i / 6)) * 0.18, 0));
  return out;
}

function addNaturalExtras() {
  const skin = new THREE.MeshStandardMaterial({ color: 0xf0b18c, roughness: 0.62, transparent: true, opacity: 0.52, side: THREE.DoubleSide });
  const shirt = new THREE.MeshStandardMaterial({ color: 0x5bd7ff, roughness: 0.58, transparent: true, opacity: 0.55 });
  if (mode === 'hand') {
    const palm = new THREE.Mesh(new THREE.SphereGeometry(0.50, 28, 16), skin);
    palm.scale.set(1.05, 0.70, 0.25);
    palm.userData.kind = 'handPalm';
    root.add(palm);
    extras.push(palm);
  } else if (mode === 'face') {
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.92, 32, 20), skin);
    head.scale.set(0.82, 1.05, 0.55);
    head.userData.kind = 'faceHead';
    root.add(head);
    extras.push(head);
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.25, transparent: true, opacity: 0.55 });
    [-0.27, 0.27].forEach(x => {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 8), eyeMat);
      eye.position.set(x, 0.13, 0.43);
      eye.userData.kind = 'faceEye';
      root.add(eye);
      extras.push(eye);
    });
  } else {
    const torso = new THREE.Mesh(new THREE.BoxGeometry(1.25, mode === 'upper' ? 1.25 : 1.45, 0.24), shirt);
    torso.userData.kind = 'torso';
    torso.position.set(0, -0.28, -0.04);
    root.add(torso);
    extras.push(torso);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.25, 24, 16), skin);
    head.userData.kind = 'poseHead';
    head.position.set(0, 1.08, 0);
    root.add(head);
    extras.push(head);
  }
}

function loadRiggedAssetInBackground() {
  if (!GLTFLoader) return;
  const token = ++assetLoadToken;
  const url = mode === 'hand' ? HAND_ASSET_URL : mode === 'face' ? FACE_ASSET_URL : BODY_ASSET_URL;
  assetStatus = 'loading';
  updateLabels();
  const loader = new GLTFLoader();
  const timeout = setTimeout(() => {
    if (token === assetLoadToken) {
      assetStatus = 'fallback';
      updateLabels();
    }
  }, 5500);
  loader.load(url, gltf => {
    if (token !== assetLoadToken) return;
    clearTimeout(timeout);
    assetRoot = gltf.scene;
    normalizeAsset(assetRoot);
    root.add(assetRoot);
    assetStatus = 'loaded';
    updateLabels();
  }, undefined, error => {
    console.warn('Rigged asset failed to load', error);
    if (token === assetLoadToken) {
      clearTimeout(timeout);
      assetStatus = 'fallback';
      updateLabels();
    }
  });
}

function normalizeAsset(object) {
  object.traverse(child => {
    if (child.isMesh || child.isSkinnedMesh) {
      child.frustumCulled = false;
      if (child.material) {
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach(material => {
          material.transparent = true;
          material.opacity = mode === 'body' || mode === 'upper' ? 0.86 : mode === 'face' ? 0.92 : 0.96;
          material.depthWrite = true;
          material.clippingPlanes = mode === 'upper' ? [new THREE.Plane(new THREE.Vector3(0, 1, 0), 0.45)] : [];
        });
      }
    }
  });
  const box = new THREE.Box3().setFromObject(object);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);
  object.position.sub(center);
  const maxDim = Math.max(size.x, size.y, size.z, 0.001);
  const targetSize = mode === 'hand' ? 2.35 : mode === 'face' ? 2.05 : mode === 'upper' ? 2.95 : 3.25;
  object.scale.setScalar(targetSize / maxDim);
  if (mode === 'hand') object.rotation.set(0, Math.PI, 0);
  if (mode === 'face') object.position.y -= 0.10;
  if (mode === 'upper') object.position.y -= 0.18;
}

function animate(now) {
  requestAnimationFrame(animate);
  if (!renderer) return;
  const dt = Math.max(1, now - lastFrame);
  fpsValue = fpsValue * 0.9 + (1000 / dt) * 0.1;
  lastFrame = now;
  if (running && tracker && !loadingTracker && el.video.readyState >= 2 && now - lastDetect > 32) {
    lastDetect = now;
    detect(now);
  }
  const follow = misses > 10 ? 0.06 : 0.42;
  for (let i = 0; i < smooth.length; i++) smooth[i].lerp(target[i], follow);
  drawModel();
  if (grid) grid.rotation.z = now * 0.000035;
  renderer.render(scene, camera3d);
  el.fps.textContent = 'FPS: ' + Math.round(fpsValue);
  el.videoSize.textContent = 'Video: ' + (el.video.videoWidth || '?') + 'x' + (el.video.videoHeight || '?');
}

function detect(now) {
  analyzed++;
  el.analyzed.textContent = 'Analyzed: ' + analyzed;
  try {
    handleResult(tracker.detectForVideo(el.video, now));
  } catch (error) {
    console.error(error);
    setStatus('Tracker error. Retrying...', 'bad');
  }
}

function handleResult(result) {
  clearOverlay();
  let landmarks = null;
  let score = 0.95;
  let labelText = label(mode) + ': tracked';
  if (mode === 'hand') {
    if (!result.landmarks || !result.landmarks.length) return noDetection('No hand detected.');
    landmarks = result.landmarks[0];
    const world = result.worldLandmarks && result.worldLandmarks[0] ? result.worldLandmarks[0] : null;
    const handInfo = result.handednesses && result.handednesses[0] && result.handednesses[0][0];
    if (handInfo) {
      labelText = 'Hand: ' + handInfo.categoryName;
      score = handInfo.score;
    }
    target = convertLandmarks(landmarks, world, 21, [0,5,9,13,17]);
    drawOverlay(landmarks, HAND_CONN, true);
  } else if (mode === 'face') {
    if (!result.faceLandmarks || !result.faceLandmarks.length) return noDetection('No face detected.');
    landmarks = result.faceLandmarks[0];
    target = FACE_IDS.map(index => mapImagePoint(landmarks[index], 5.0, 5.8));
    centerAndScale(target, [4,6,8,10,11]);
    if (result.faceBlendshapes && result.faceBlendshapes[0] && result.faceBlendshapes[0].categories) {
      const eyeBlink = result.faceBlendshapes[0].categories.find(item => item.categoryName === 'eyeBlinkLeft');
      if (eyeBlink) score = Math.max(0.5, 1 - eyeBlink.score * 0.25);
    }
    drawFaceOverlay(landmarks);
  } else {
    if (!result.landmarks || !result.landmarks.length) return noDetection(mode === 'upper' ? 'No upper body detected.' : 'No body detected. Step back.');
    landmarks = result.landmarks[0];
    const world = result.worldLandmarks && result.worldLandmarks[0] ? result.worldLandmarks[0] : null;
    target = convertLandmarks(landmarks, world, 33, [11,12,23,24]);
    drawOverlay(landmarks, mode === 'upper' ? UPPER_CONN : BODY_CONN, false);
  }
  gotDetection(labelText, score);
}

function noDetection(message) {
  misses++;
  if (misses > 8) {
    setStatus(message, 'bad');
    el.noHand.textContent = message;
    el.noHand.classList.add('show');
    el.handedness.textContent = label(mode) + ': none';
    el.confidence.textContent = 'Confidence: 0%';
  }
  if (!triedRelaxed && analyzed > 100 && tracked === 0) {
    triedRelaxed = true;
    initTracker(true).catch(console.warn);
  }
}

function gotDetection(name, score) {
  misses = 0;
  tracked++;
  el.tracked.textContent = 'Tracked: ' + tracked;
  el.noHand.classList.remove('show');
  el.handedness.textContent = name;
  el.confidence.textContent = 'Confidence: ' + Math.round(score * 100) + '%';
  setStatus(name.replace(':', '') + ' detected.', 'good');
}

function convertLandmarks(imageLandmarks, worldLandmarks, count, centerIndices) {
  let out;
  if (worldLandmarks && worldLandmarks.length >= count) out = worldLandmarks.slice(0, count).map(point => new THREE.Vector3(-point.x * 4.2, -point.y * 4.2, -point.z * 4.2));
  else out = imageLandmarks.slice(0, count).map(point => mapImagePoint(point, mode === 'body' || mode === 'upper' ? 5.8 : 5.3, mode === 'body' || mode === 'upper' ? 6.4 : 5.8));
  centerAndScale(out, centerIndices);
  return out;
}

function mapImagePoint(point, width, height) {
  if (!point) return new THREE.Vector3();
  return new THREE.Vector3((0.5 - point.x) * width, (0.5 - point.y) * height, -(point.z || 0) * 4.0);
}

function centerAndScale(array, centerIndices) {
  const center = new THREE.Vector3();
  let n = 0;
  centerIndices.forEach(index => {
    if (array[index]) {
      center.add(array[index]);
      n++;
    }
  });
  center.multiplyScalar(1 / Math.max(1, n));
  array.forEach(point => point.sub(center));
  let ref = 1;
  if ((mode === 'body' || mode === 'upper') && array[11] && array[12]) ref = distance(array[11], array[12]);
  else if (mode === 'hand' && array[0] && array[9]) ref = distance(array[0], array[9]);
  else if (mode === 'face' && array[2] && array[3]) ref = distance(array[2], array[3]);
  const scale = THREE.MathUtils.clamp((mode === 'body' || mode === 'upper' ? 1.55 : 1.22) / Math.max(ref, 0.1), 0.55, 2.85);
  array.forEach(point => point.multiplyScalar(scale));
}

function drawModel() {
  const showGuide = modelStyle === 'stick' || !assetRoot;
  for (const mesh of lines) {
    const a = smooth[mesh.userData.a];
    const b = smooth[mesh.userData.b];
    mesh.visible = showGuide;
    if (a && b) placeCylinder(mesh, a, b, radiusFor(mesh.userData.a, mesh.userData.b));
  }
  for (const joint of joints) {
    const point = smooth[joint.userData.index];
    joint.visible = showGuide && !(mode === 'upper' && isLowerBodyIndex(joint.userData.index));
    if (point) {
      joint.position.copy(point);
      joint.scale.setScalar(modelStyle === 'stick' ? 0.045 : mode === 'body' || mode === 'upper' ? 0.075 : 0.085);
    }
  }
  for (const extra of extras) extra.visible = modelStyle === 'natural' && !assetRoot;
  updateExtras();
  updateAssetPose();
}

function updateExtras() {
  for (const extra of extras) {
    if (extra.userData.kind === 'handPalm' && smooth[0] && smooth[9]) {
      extra.position.copy(midpoint([smooth[0], smooth[5], smooth[9], smooth[13], smooth[17]]));
      extra.scale.set(1.05, 0.70, 0.25);
    } else if (extra.userData.kind === 'faceHead' && smooth[4]) {
      extra.position.copy(smooth[4]);
      extra.position.z -= 0.08;
    } else if (extra.userData.kind === 'torso' && smooth[11] && smooth[12] && smooth[23] && smooth[24]) {
      extra.position.copy(midpoint([smooth[11], smooth[12], smooth[23], smooth[24]]));
    } else if (extra.userData.kind === 'poseHead' && smooth[0]) {
      extra.position.copy(smooth[0]);
    }
  }
}

function updateAssetPose() {
  if (!assetRoot) return;
  let center = null;
  if (mode === 'hand') center = midpoint([smooth[0], smooth[5], smooth[9], smooth[13], smooth[17]]);
  else if (mode === 'face') center = smooth[4] ? smooth[4].clone() : midpoint(smooth);
  else if (smooth[11] && smooth[12] && smooth[23] && smooth[24]) center = midpoint([smooth[11], smooth[12], smooth[23], smooth[24]]);
  if (!center) center = new THREE.Vector3();
  if (!lastAssetCenter) lastAssetCenter = center.clone();
  lastAssetCenter.lerp(center, 0.20);
  const targetOffset = mode === 'hand' ? new THREE.Vector3(0, 0, 0) : mode === 'face' ? new THREE.Vector3(0, -0.18, -0.08) : mode === 'upper' ? new THREE.Vector3(0, -0.14, -0.02) : new THREE.Vector3(0, -0.28, -0.02);
  assetRoot.position.copy(lastAssetCenter).add(targetOffset);
  if ((mode === 'body' || mode === 'upper') && smooth[11] && smooth[12]) {
    const shoulderTilt = smooth[12].y - smooth[11].y;
    assetRoot.rotation.z = THREE.MathUtils.clamp(shoulderTilt * 0.18, -0.35, 0.35);
  } else if (mode === 'face' && smooth[2] && smooth[3]) {
    assetRoot.rotation.y = THREE.MathUtils.clamp((smooth[3].z - smooth[2].z) * 0.15, -0.3, 0.3);
  } else if (mode === 'hand' && smooth[0] && smooth[9]) {
    const angle = Math.atan2(smooth[9].x - smooth[0].x, smooth[9].y - smooth[0].y);
    assetRoot.rotation.z = -angle * 0.15;
  }
}

function midpoint(items) {
  const out = new THREE.Vector3();
  let n = 0;
  items.forEach(item => {
    if (item) {
      out.add(item);
      n++;
    }
  });
  return out.multiplyScalar(1 / Math.max(1, n));
}

function isLowerBodyIndex(index) {
  return index === 25 || index === 26 || index === 27 || index === 28 || index === 29 || index === 30 || index === 31 || index === 32;
}

function radiusFor(a, b) {
  const base = modelStyle === 'stick' ? 0.018 : mode === 'body' || mode === 'upper' ? 0.065 : 0.055;
  return base * (mode === 'hand' && (TIP.has(a) || TIP.has(b)) ? 0.72 : 1);
}

function placeCylinder(mesh, a, b, radius) {
  const direction = v1.subVectors(b, a);
  const length = Math.max(direction.length(), 0.0001);
  mesh.position.copy(v2.copy(a).add(b).multiplyScalar(0.5));
  mesh.quaternion.setFromUnitVectors(yAxis, direction.normalize());
  mesh.scale.set(radius, length, radius);
  if (mode === 'upper' && (isLowerBodyIndex(mesh.userData.a) || isLowerBodyIndex(mesh.userData.b))) mesh.visible = false;
}

function drawOverlay(landmarks, connections, handTips) {
  const w = el.overlay.width;
  const h = el.overlay.height;
  overlayCtx.save();
  overlayCtx.lineCap = 'round';
  connections.forEach(pair => {
    const a = landmarks[pair[0]];
    const b = landmarks[pair[1]];
    if (!a || !b) return;
    overlayCtx.beginPath();
    overlayCtx.moveTo(a.x * w, a.y * h);
    overlayCtx.lineTo(b.x * w, b.y * h);
    overlayCtx.lineWidth = 5;
    overlayCtx.strokeStyle = 'rgba(110,214,255,.92)';
    overlayCtx.stroke();
    overlayCtx.lineWidth = 1.5;
    overlayCtx.strokeStyle = 'rgba(0,0,0,.55)';
    overlayCtx.stroke();
  });
  landmarks.forEach((point, index) => {
    if (!point) return;
    if (mode === 'upper' && isLowerBodyIndex(index)) return;
    overlayCtx.beginPath();
    overlayCtx.arc(point.x * w, point.y * h, handTips && TIP.has(index) ? 7 : 4, 0, Math.PI * 2);
    overlayCtx.fillStyle = handTips && TIP.has(index) ? '#90f4ff' : '#fff';
    overlayCtx.fill();
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeStyle = 'rgba(0,0,0,.62)';
    overlayCtx.stroke();
  });
  overlayCtx.restore();
}

function drawFaceOverlay(landmarks) {
  const w = el.overlay.width;
  const h = el.overlay.height;
  overlayCtx.save();
  overlayCtx.lineCap = 'round';
  FACE_CONN.forEach(pair => {
    const a = landmarks[FACE_IDS[pair[0]]];
    const b = landmarks[FACE_IDS[pair[1]]];
    if (!a || !b) return;
    overlayCtx.beginPath();
    overlayCtx.moveTo(a.x * w, a.y * h);
    overlayCtx.lineTo(b.x * w, b.y * h);
    overlayCtx.lineWidth = 4;
    overlayCtx.strokeStyle = 'rgba(110,214,255,.92)';
    overlayCtx.stroke();
  });
  FACE_IDS.forEach(index => {
    const point = landmarks[index];
    if (!point) return;
    overlayCtx.beginPath();
    overlayCtx.arc(point.x * w, point.y * h, 4.5, 0, Math.PI * 2);
    overlayCtx.fillStyle = '#fff';
    overlayCtx.fill();
    overlayCtx.strokeStyle = 'rgba(0,0,0,.62)';
    overlayCtx.stroke();
  });
  overlayCtx.restore();
}

function clearOverlay() {
  resizeOverlay();
  overlayCtx.clearRect(0, 0, el.overlay.width, el.overlay.height);
}

function makeGrid() {
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x66d9ff, transparent: true, opacity: 0.09 });
  const ringMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.055 });
  for (let radius = 1.2; radius <= 6; radius += 0.8) {
    const vertices = [];
    for (let i = 0; i <= 128; i++) {
      const angle = i / 128 * Math.PI * 2;
      vertices.push(new THREE.Vector3(Math.cos(angle) * radius, Math.sin(angle) * radius, -3.2));
    }
    grid.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(vertices), ringMaterial));
  }
  for (let i = 0; i < 18; i++) {
    const angle = i / 18 * Math.PI * 2;
    grid.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(Math.cos(angle) * 0.55, Math.sin(angle) * 0.55, -3.2),
      new THREE.Vector3(Math.cos(angle) * 6.5, Math.sin(angle) * 6.5, -3.2)
    ]), lineMaterial));
  }
}

function resizeOverlay() {
  const rect = el.overlay.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (el.overlay.width !== width || el.overlay.height !== height) {
    el.overlay.width = width;
    el.overlay.height = height;
    overlayCtx = el.overlay.getContext('2d', { alpha: true });
  }
}

function resizeAll() {
  resizeOverlay();
  if (!renderer || !camera3d) return;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera3d.aspect = window.innerWidth / window.innerHeight;
  camera3d.updateProjectionMatrix();
}

function setStatus(text, state) {
  el.statusText.textContent = text;
  el.stateDot.classList.toggle('good', state === 'good');
  el.stateDot.classList.toggle('bad', state === 'bad');
}

function showLog(text) {
  el.startLog.textContent = text;
  el.startLog.style.display = 'block';
}

function friendlyError(error) {
  const details = '\n\nTechnical details: ' + (error.name || 'Error') + ': ' + (error.message || String(error));
  if (!window.isSecureContext) return 'Camera access needs HTTPS or localhost.' + details;
  if (/permission|denied|NotAllowed/i.test((error.name || '') + ' ' + (error.message || ''))) return 'Chrome denied camera access. Allow camera for this site and try again.' + details;
  return 'Something failed while starting the camera, tracker, or 3D view.' + details;
}

function distance(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}
