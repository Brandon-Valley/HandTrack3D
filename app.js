'use strict';

const VERSION = '2026-06-27-no-blocking-asset';
const MEDIAPIPE_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const MEDIAPIPE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17], [5, 17]
];
const PALM_TRIANGLES = [0, 5, 9, 0, 9, 13, 0, 13, 17, 5, 9, 13, 5, 13, 17];
const TIP = new Set([4, 8, 12, 16, 20]);
const MCP = new Set([5, 9, 13, 17]);

let THREE = null;
let FilesetResolver = null;
let HandLandmarker = null;
let handLandmarker = null;
let renderer = null;
let scene = null;
let camera3d = null;
let handGroup = null;
let gridGroup = null;
let palmGeometry = null;
let palmMesh = null;
let segmentMeshes = [];
let jointMeshes = [];
let nailMeshes = [];
let defaultPoints = [];
let targetPoints = [];
let smoothPoints = [];
let targetPosition = null;
let smoothPosition = null;
let targetRotation = null;
let smoothRotation = null;
let yAxis = null;
let tempA = null;
let tempB = null;
let tempC = null;
let tempD = null;
let running = false;
let sceneReady = false;
let analyzedFrames = 0;
let trackedFrames = 0;
let noHandFrames = 0;
let lastDetectMs = 0;
let lastFrameMs = performance.now();
let fpsValue = 0;
let triedRelaxedTracker = false;

const el = {
  stage: document.getElementById('stage'),
  fallback: document.getElementById('stageFallback'),
  video: document.getElementById('video'),
  overlay: document.getElementById('overlay'),
  startPanel: document.getElementById('startPanel'),
  startButton: document.getElementById('startButton'),
  startLog: document.getElementById('startLog'),
  startError: document.getElementById('startError'),
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

el.startButton.addEventListener('click', startDemo);
window.addEventListener('resize', resizeAll);
window.addEventListener('orientationchange', () => setTimeout(resizeAll, 250));
resizeOverlay();
showStartupNote((window.isSecureContext ? 'Ready.' : 'HTTPS is required.') + ' Version: ' + VERSION);
el.tracker.textContent = 'Tracker: waiting';
el.model.textContent = 'Model: built-in hand rig';

async function startDemo() {
  el.startButton.disabled = true;
  el.startButton.textContent = 'Starting camera...';
  el.startError.textContent = '';
  el.startError.style.display = 'none';
  setStatus('Requesting front camera...', 'warn');
  showStartupNote('Requesting camera first. This build no longer blocks startup on a remote hand asset.');

  try {
    await startCamera();
    el.videoSize.textContent = 'Video: ' + (el.video.videoWidth || '?') + 'x' + (el.video.videoHeight || '?');
    el.startButton.textContent = 'Loading libraries...';
    setStatus('Camera on. Loading libraries...', 'warn');
    await loadLibraries();
    setupScene();
    el.startButton.textContent = 'Loading tracker...';
    setStatus('Loading hand tracker...', 'warn');
    await initTracker(false);
    running = true;
    el.startPanel.style.display = 'none';
    setStatus('Tracking is running. Show one open hand.', 'warn');
  } catch (error) {
    console.error(error);
    running = false;
    stopCamera();
    el.startButton.disabled = false;
    el.startButton.textContent = 'Try again';
    setStatus('Could not start.', 'bad');
    el.startError.textContent = friendlyError(error);
    el.startError.style.display = 'block';
    showStartupNote('Startup stopped. The exact error is below.');
  }
}

async function startCamera() {
  if (!window.isSecureContext) throw new Error('Camera access needs HTTPS or localhost.');
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('getUserMedia is not available in this browser.');

  const attempts = [
    { audio: false, video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 60 } } },
    { audio: false, video: { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 540 } } },
    { audio: false, video: true }
  ];

  let lastError = null;
  for (let i = 0; i < attempts.length; i++) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia(attempts[i]);
      el.video.srcObject = stream;
      el.video.setAttribute('playsinline', '');
      el.video.muted = true;
      await waitForVideo(el.video, 9000);
      await el.video.play();
      resizeOverlay();
      return;
    } catch (error) {
      lastError = error;
      stopCamera();
      showStartupNote('Camera attempt ' + (i + 1) + ' failed. Trying fallback camera settings...');
    }
  }
  throw lastError || new Error('No usable camera stream was found.');
}

function waitForVideo(video, timeoutMs) {
  return new Promise((resolve, reject) => {
    if (video.videoWidth > 0 && video.videoHeight > 0 && video.readyState >= 2) {
      resolve();
      return;
    }
    let done = false;
    const timer = setTimeout(() => {
      if (done) return;
      done = true;
      cleanup();
      reject(new Error('Camera opened, but video did not become ready.'));
    }, timeoutMs);
    function cleanup() {
      clearTimeout(timer);
      video.removeEventListener('loadedmetadata', ready);
      video.removeEventListener('canplay', ready);
      video.removeEventListener('playing', ready);
      video.removeEventListener('error', fail);
    }
    function ready() {
      if (done) return;
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        done = true;
        cleanup();
        resolve();
      }
    }
    function fail() {
      if (done) return;
      done = true;
      cleanup();
      reject(new Error('Video element reported an error.'));
    }
    video.addEventListener('loadedmetadata', ready);
    video.addEventListener('canplay', ready);
    video.addEventListener('playing', ready);
    video.addEventListener('error', fail);
    video.play().catch(() => {});
  });
}

function stopCamera() {
  if (el.video.srcObject && el.video.srcObject.getTracks) {
    el.video.srcObject.getTracks().forEach(track => track.stop());
  }
  el.video.srcObject = null;
}

async function loadLibraries() {
  const modules = await Promise.all([
    import('three'),
    import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs')
  ]);
  THREE = modules[0];
  FilesetResolver = modules[1].FilesetResolver;
  HandLandmarker = modules[1].HandLandmarker;
}

function setupScene() {
  if (sceneReady) return;
  yAxis = new THREE.Vector3(0, 1, 0);
  tempA = new THREE.Vector3();
  tempB = new THREE.Vector3();
  tempC = new THREE.Vector3();
  tempD = new THREE.Vector3();
  targetPosition = new THREE.Vector3(0, -0.1, 0);
  smoothPosition = targetPosition.clone();
  targetRotation = new THREE.Quaternion();
  smoothRotation = new THREE.Quaternion();
  defaultPoints = makeDefaultPoints();
  targetPoints = defaultPoints.map(point => point.clone());
  smoothPoints = defaultPoints.map(point => point.clone());

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
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

  handGroup = new THREE.Group();
  scene.add(handGroup);
  gridGroup = new THREE.Group();
  scene.add(gridGroup);
  makeGrid();
  buildHandRig();
  applyPoints(defaultPoints, true);
  sceneReady = true;
  requestAnimationFrame(animate);
}

function makeDefaultPoints() {
  const source = [
    [0, -1.18, 0], [-0.56, -0.78, 0.05], [-0.88, -0.36, 0.08], [-1.08, 0.04, 0.07], [-1.26, 0.42, 0.03],
    [-0.48, -0.24, 0.02], [-0.58, 0.38, 0.06], [-0.64, 0.92, 0.04], [-0.68, 1.38, 0],
    [-0.12, -0.12, 0.02], [-0.14, 0.58, 0.08], [-0.16, 1.22, 0.06], [-0.18, 1.78, 0.02],
    [0.26, -0.2, 0.02], [0.34, 0.42, 0.07], [0.4, 0.98, 0.05], [0.44, 1.46, 0.02],
    [0.6, -0.36, 0], [0.78, 0.16, 0.05], [0.9, 0.62, 0.03], [1, 1.02, 0]
  ];
  return source.map(point => new THREE.Vector3(point[0], point[1], point[2]));
}

function buildHandRig() {
  const skin = new THREE.MeshStandardMaterial({ color: 0xe9a47b, roughness: 0.58, metalness: 0.02 });
  const palmMaterial = new THREE.MeshStandardMaterial({ color: 0xf0b18c, roughness: 0.64, transparent: true, opacity: 0.78, side: THREE.DoubleSide });
  const jointMaterial = new THREE.MeshStandardMaterial({ color: 0xffd1b8, roughness: 0.48 });
  const tipMaterial = new THREE.MeshStandardMaterial({ color: 0xffbf9f, roughness: 0.42 });
  const nailMaterial = new THREE.MeshStandardMaterial({ color: 0xffeee8, roughness: 0.34 });
  const knuckleMaterial = new THREE.MeshStandardMaterial({ color: 0xf6b28f, roughness: 0.46 });
  const cylinder = new THREE.CylinderGeometry(1, 1, 1, 22, 1, false);
  const sphere = new THREE.SphereGeometry(1, 24, 14);

  segmentMeshes = CONNECTIONS.slice(0, 20).map(pair => {
    const mesh = new THREE.Mesh(cylinder, skin);
    mesh.userData = { a: pair[0], b: pair[1] };
    handGroup.add(mesh);
    return mesh;
  });

  jointMeshes = defaultPoints.map((_, index) => {
    const material = TIP.has(index) ? tipMaterial : MCP.has(index) ? knuckleMaterial : jointMaterial;
    const mesh = new THREE.Mesh(sphere, material);
    handGroup.add(mesh);
    return mesh;
  });

  [4, 8, 12, 16, 20].forEach(index => {
    const nail = new THREE.Mesh(sphere, nailMaterial);
    nail.userData.index = index;
    handGroup.add(nail);
    nailMeshes.push(nail);
  });

  palmGeometry = new THREE.BufferGeometry();
  palmGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(PALM_TRIANGLES.length * 3), 3));
  palmMesh = new THREE.Mesh(palmGeometry, palmMaterial);
  handGroup.add(palmMesh);
}

async function initTracker(relaxed) {
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
  const options = {
    baseOptions: { modelAssetPath: MEDIAPIPE_MODEL_URL, delegate: relaxed ? 'CPU' : 'GPU' },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: relaxed ? 0.12 : 0.25,
    minHandPresenceConfidence: relaxed ? 0.12 : 0.25,
    minTrackingConfidence: relaxed ? 0.12 : 0.25
  };
  try {
    if (handLandmarker && handLandmarker.close) handLandmarker.close();
  } catch (error) {}
  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, options);
    el.tracker.textContent = 'Tracker: ' + (relaxed ? 'CPU relaxed' : 'GPU low threshold');
  } catch (error) {
    options.baseOptions.delegate = 'CPU';
    handLandmarker = await HandLandmarker.createFromOptions(vision, options);
    el.tracker.textContent = 'Tracker: ' + (relaxed ? 'CPU relaxed' : 'CPU low threshold');
  }
}

function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.max(1, now - lastFrameMs);
  fpsValue = fpsValue * 0.9 + (1000 / dt) * 0.1;
  lastFrameMs = now;
  if (running && handLandmarker && el.video.readyState >= 2 && now - lastDetectMs >= 30) {
    lastDetectMs = now;
    detect(now);
  }
  const follow = noHandFrames > 10 ? 0.065 : 0.44;
  for (let i = 0; i < smoothPoints.length; i++) smoothPoints[i].lerp(targetPoints[i], follow);
  smoothPosition.lerp(targetPosition, 0.26);
  smoothRotation.slerp(targetRotation, 0.24);
  handGroup.position.copy(smoothPosition);
  handGroup.quaternion.copy(smoothRotation);
  applyPoints(smoothPoints, false);
  gridGroup.rotation.z = now * 0.000035;
  renderer.render(scene, camera3d);
  el.fps.textContent = 'FPS: ' + Math.round(fpsValue);
  el.videoSize.textContent = 'Video: ' + (el.video.videoWidth || '?') + 'x' + (el.video.videoHeight || '?');
}

function detect(now) {
  analyzedFrames++;
  el.analyzed.textContent = 'Analyzed: ' + analyzedFrames;
  try {
    handleResult(handLandmarker.detectForVideo(el.video, now));
  } catch (error) {
    console.error(error);
    setStatus('Tracker error. Retrying...', 'bad');
  }
}

function handleResult(result) {
  resizeOverlay();
  overlayCtx.clearRect(0, 0, el.overlay.width, el.overlay.height);
  if (!result || !result.landmarks || result.landmarks.length === 0) {
    noHandFrames++;
    if (noHandFrames > 8) {
      targetPoints = defaultPoints.map(point => point.clone());
      targetPosition.set(0, -0.1, 0);
      targetRotation.identity();
      setStatus('No hand detected. Move hand fully into camera box.', 'bad');
      el.noHand.classList.add('show');
      el.handedness.textContent = 'Hand: none';
      el.confidence.textContent = 'Confidence: 0%';
    }
    if (!triedRelaxedTracker && analyzedFrames > 100 && trackedFrames === 0) {
      triedRelaxedTracker = true;
      setStatus('No hand yet. Switching to relaxed CPU tracker...', 'warn');
      initTracker(true).catch(console.warn);
    }
    return;
  }
  noHandFrames = 0;
  trackedFrames++;
  el.tracked.textContent = 'Tracked: ' + trackedFrames;
  el.noHand.classList.remove('show');
  const landmarks = result.landmarks[0];
  const world = result.worldLandmarks && result.worldLandmarks[0] ? result.worldLandmarks[0] : null;
  const handInfo = result.handednesses && result.handednesses[0] && result.handednesses[0][0];
  el.handedness.textContent = 'Hand: ' + (handInfo ? handInfo.categoryName : 'tracked');
  el.confidence.textContent = 'Confidence: ' + Math.round((handInfo ? handInfo.score : 0) * 100) + '%';
  setStatus('Hand detected. Animating 3D hand.', 'good');
  drawLandmarks(landmarks);
  targetPoints = convertToThreePoints(landmarks, world);
  updateHandPosition(landmarks);
  updateHandRotation(targetPoints);
}

function convertToThreePoints(imageLandmarks, worldLandmarks) {
  let points;
  if (worldLandmarks && worldLandmarks.length === 21) {
    points = worldLandmarks.map(point => new THREE.Vector3(-point.x * 16, -point.y * 16, -point.z * 16));
  } else {
    const aspect = (el.video.videoWidth || 1280) / Math.max(1, el.video.videoHeight || 720);
    const viewWidth = 5.4 * Math.min(aspect, 1.9);
    points = imageLandmarks.map(point => new THREE.Vector3((0.5 - point.x) * viewWidth, (0.5 - point.y) * 5.7, -point.z * 8.5));
  }
  const center = new THREE.Vector3();
  [0, 5, 9, 13, 17].forEach(index => center.add(points[index]));
  center.multiplyScalar(0.2);
  const scale = THREE.MathUtils.clamp(1.2 / Math.max(distance(points[0], points[9]), 0.08), 0.72, 2.25);
  return points.map(point => point.sub(center).multiplyScalar(scale));
}

function updateHandPosition(landmarks) {
  let centerX = 0;
  let centerY = 0;
  let minX = 1;
  let maxX = 0;
  for (const point of landmarks) {
    centerX += point.x;
    centerY += point.y;
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
  }
  centerX /= landmarks.length;
  centerY /= landmarks.length;
  const z = THREE.MathUtils.clamp((0.28 - Math.max(0.04, maxX - minX)) * 7, -1.15, 1.55);
  targetPosition.set((0.5 - centerX) * 3.8, (0.5 - centerY) * 2.55 - 0.1, z);
}

function updateHandRotation(points) {
  const x = tempA.subVectors(points[17], points[5]).normalize();
  const y = tempB.subVectors(points[9], points[0]).normalize();
  const z = tempC.crossVectors(x, y).normalize();
  if (z.lengthSq() < 0.001) return;
  y.crossVectors(z, x).normalize();
  targetRotation.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
  targetRotation.slerp(new THREE.Quaternion(), 0.35);
}

function applyPoints(points, snap) {
  const working = snap ? points.map(point => point.clone()) : points;
  for (const mesh of segmentMeshes) placeCylinder(mesh, working[mesh.userData.a], working[mesh.userData.b], radiusFor(mesh.userData.a, mesh.userData.b));
  for (let i = 0; i < jointMeshes.length; i++) {
    jointMeshes[i].position.copy(working[i]);
    jointMeshes[i].scale.setScalar(TIP.has(i) ? 0.083 : MCP.has(i) ? 0.096 : i === 0 ? 0.11 : 0.069);
  }
  for (const nail of nailMeshes) {
    const index = nail.userData.index;
    nail.position.copy(working[index]);
    tempD.subVectors(working[index], working[index - 1]).normalize();
    nail.position.addScaledVector(tempD, 0.028);
    nail.quaternion.setFromUnitVectors(yAxis, tempD);
    nail.scale.set(0.055, 0.025, 0.088);
  }
  const array = palmGeometry.attributes.position.array;
  for (let i = 0; i < PALM_TRIANGLES.length; i++) {
    const point = working[PALM_TRIANGLES[i]];
    array[i * 3 + 0] = point.x;
    array[i * 3 + 1] = point.y;
    array[i * 3 + 2] = point.z - 0.02;
  }
  palmGeometry.attributes.position.needsUpdate = true;
  palmGeometry.computeVertexNormals();
}

function placeCylinder(mesh, a, b, radius) {
  const direction = tempA.subVectors(b, a);
  const length = Math.max(direction.length(), 0.0001);
  mesh.position.copy(tempB.copy(a).add(b).multiplyScalar(0.5));
  mesh.quaternion.setFromUnitVectors(yAxis, direction.normalize());
  mesh.scale.set(radius, length, radius);
}

function radiusFor(a, b) {
  if (a === 0 || b === 0) return 0.07;
  if (MCP.has(a) || MCP.has(b)) return 0.063;
  if (TIP.has(a) || TIP.has(b)) return 0.044;
  return 0.053;
}

function drawLandmarks(landmarks) {
  const w = el.overlay.width;
  const h = el.overlay.height;
  overlayCtx.save();
  overlayCtx.lineCap = 'round';
  overlayCtx.lineJoin = 'round';
  for (const pair of CONNECTIONS) {
    const a = landmarks[pair[0]];
    const b = landmarks[pair[1]];
    overlayCtx.beginPath();
    overlayCtx.moveTo(a.x * w, a.y * h);
    overlayCtx.lineTo(b.x * w, b.y * h);
    overlayCtx.lineWidth = 5;
    overlayCtx.strokeStyle = 'rgba(110,214,255,.92)';
    overlayCtx.stroke();
    overlayCtx.lineWidth = 1.5;
    overlayCtx.strokeStyle = 'rgba(0,0,0,.55)';
    overlayCtx.stroke();
  }
  for (let i = 0; i < landmarks.length; i++) {
    const point = landmarks[i];
    overlayCtx.beginPath();
    overlayCtx.arc(point.x * w, point.y * h, TIP.has(i) ? 7 : 4.5, 0, Math.PI * 2);
    overlayCtx.fillStyle = TIP.has(i) ? '#90f4ff' : '#fff';
    overlayCtx.fill();
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeStyle = 'rgba(0,0,0,.62)';
    overlayCtx.stroke();
  }
  overlayCtx.restore();
}

function makeGrid() {
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x66d9ff, transparent: true, opacity: 0.09 });
  const ringMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.055 });
  for (let radius = 1.2; radius <= 6; radius += 0.8) {
    const points = [];
    for (let i = 0; i <= 128; i++) {
      const angle = i / 128 * Math.PI * 2;
      points.push(new THREE.Vector3(Math.cos(angle) * radius, Math.sin(angle) * radius, -3.2));
    }
    gridGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), ringMaterial));
  }
  for (let i = 0; i < 18; i++) {
    const angle = i / 18 * Math.PI * 2;
    gridGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
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

function setStatus(text, mode) {
  el.statusText.textContent = text;
  el.stateDot.classList.toggle('good', mode === 'good');
  el.stateDot.classList.toggle('bad', mode === 'bad');
}

function showStartupNote(text) {
  el.startLog.textContent = text;
  el.startLog.style.display = 'block';
}

function friendlyError(error) {
  const name = error && error.name ? error.name : 'Error';
  const message = error && error.message ? error.message : String(error);
  const details = '\n\nTechnical details: ' + name + ': ' + message;
  if (!window.isSecureContext) return 'Camera access is blocked because this page is not running from HTTPS or localhost.' + details;
  if (/NotAllowedError|Permission|denied|permission/i.test(name + ' ' + message)) return "Chrome denied camera access. Change this site's camera setting to Allow and check Android Settings > Apps > Chrome > Permissions > Camera." + details;
  if (/NotFoundError|DevicesNotFoundError|OverconstrainedError|Constraint|facingMode/i.test(name + ' ' + message)) return 'I could not find a usable front camera. I tried front camera first, then any camera.' + details;
  if (/NotReadableError|TrackStartError|Could not start video source/i.test(name + ' ' + message)) return 'The browser found the camera but could not start it. Close other apps or tabs using the camera and try again.' + details;
  if (/import|module|Failed to fetch|Load failed|cdn|network|ERR_/i.test(message)) return 'A library or model file failed to load. Check internet access or browser restrictions.' + details;
  return 'Something failed while starting the camera, renderer, or tracker.' + details;
}

function distance(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}
