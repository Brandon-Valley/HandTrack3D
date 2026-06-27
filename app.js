'use strict';

const VERSION = '2026-06-27-modes-simple';
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task';
const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';

const HAND_CONN = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],[5,17]];
const POSE_CONN = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,31],[28,32],[0,11],[0,12]];
const FACE_IDS = [10,152,234,454,1,4,33,133,263,362,61,291,13,14,0,17,50,280];
const FACE_CONN = [[0,2],[0,3],[2,1],[3,1],[6,7],[8,9],[10,12],[12,11],[11,13],[13,10],[4,5],[14,15]];
const TIP = new Set([4,8,12,16,20]);

let THREE, FilesetResolver, HandLandmarker, FaceLandmarker, PoseLandmarker, vision;
let renderer, scene, camera3d, root, grid, tracker, trackerMode = '';
let mode = 'hand', style = 'natural', running = false, loadingTracker = false;
let points = [], target = [], smooth = [], connections = [], lastDetect = 0, lastFrame = performance.now(), fps = 0;
let analyzed = 0, tracked = 0, misses = 0, triedRelaxed = false;
let yAxis, tmp1, tmp2;
let meshes = { lines: [], joints: [], extras: [] };

const el = {
  app: document.getElementById('app'), stage: document.getElementById('stage'), fallback: document.getElementById('stageFallback'),
  video: document.getElementById('video'), overlay: document.getElementById('overlay'), startPanel: document.getElementById('startPanel'),
  startButton: document.getElementById('startButton'), startLog: document.getElementById('startLog'), startError: document.getElementById('startError'),
  statusCard: document.getElementById('statusCard'), statusText: document.getElementById('statusText'), stateDot: document.getElementById('stateDot'),
  handedness: document.getElementById('handedness'), confidence: document.getElementById('confidence'), fps: document.getElementById('fps'),
  tracked: document.getElementById('trackedFrames'), analyzed: document.getElementById('analyzedFrames'), tracker: document.getElementById('trackerMode'),
  videoSize: document.getElementById('videoSize'), model: document.getElementById('modelMode'), noHand: document.getElementById('noHandBanner')
};
let ctx = el.overlay.getContext('2d', { alpha: true });

setupStatusPanel();
setupModePanel();
resizeOverlay();
showLog('Ready. Version: ' + VERSION);
updateLabels();
el.startButton.addEventListener('click', startDemo);
window.addEventListener('resize', resizeAll);
window.addEventListener('orientationchange', () => setTimeout(resizeAll, 250));

function setupStatusPanel() {
  const css = document.createElement('style');
  css.textContent = `
    #statusCard{transition:all .16s ease}#statusToggle{appearance:none;width:100%;min-height:0;margin:9px 0 0;padding:7px 9px;border-radius:999px;border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.08);color:var(--text);font-size:11px;font-weight:950;box-shadow:none}
    #statusCard.collapsed{width:auto;min-width:0;max-width:calc(100vw - 88px);padding:8px 9px;border-radius:999px;display:flex;align-items:center;gap:8px}#statusCard.collapsed #title{margin:0;font-size:13px;line-height:1;white-space:nowrap}#statusCard.collapsed .statusLine{font-size:0;gap:0}#statusCard.collapsed #statusText,#statusCard.collapsed #metrics{display:none}#statusCard.collapsed #statusToggle{width:auto;margin:0;padding:6px 8px;font-size:11px;background:rgba(255,255,255,.1)}
    #modePanel{position:absolute;z-index:7;top:max(12px,env(safe-area-inset-top));left:50%;transform:translateX(-50%);width:min(360px,calc(100vw - 108px));pointer-events:auto}#modeToggle{width:100%;min-height:42px;padding:10px 13px;border:1px solid rgba(255,255,255,.22);border-radius:999px;background:rgba(8,12,22,.9);color:#fff;box-shadow:0 18px 54px rgba(0,0,0,.36);backdrop-filter:blur(18px);font-size:13px}#modeSheet{display:none;margin-top:8px;padding:10px;border:1px solid rgba(255,255,255,.18);border-radius:18px;background:rgba(8,12,22,.94);box-shadow:0 18px 54px rgba(0,0,0,.42);backdrop-filter:blur(18px)}#modePanel.open #modeSheet{display:block}.controlLabel{margin:0 0 7px;color:var(--muted);font-size:11px;font-weight:900;text-transform:uppercase;letter-spacing:.08em}.buttonRow{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:10px}.buttonRow.two{grid-template-columns:repeat(2,1fr);margin-bottom:0}.modeChoice{min-height:38px;padding:9px 8px;border-radius:999px;border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.07);color:var(--text);box-shadow:none;font-size:12px}.modeChoice.active{background:linear-gradient(180deg,#e4fbff,#65d9ff);color:#03131d;border-color:transparent}
    @media(max-width:560px){#statusCard.collapsed{max-width:calc(100vw - 84px)}#modePanel{top:calc(max(12px,env(safe-area-inset-top)) + 52px);left:max(12px,env(safe-area-inset-left));right:auto;transform:none;width:min(250px,calc(100vw - 96px))}#modeToggle{min-height:38px;font-size:12px;padding:9px 11px}}
  `;
  document.head.appendChild(css);
  const button = document.createElement('button');
  button.id = 'statusToggle';
  button.type = 'button';
  button.textContent = 'Stats';
  el.statusCard.insertBefore(button, document.getElementById('metrics'));
  el.statusCard.classList.add('collapsed');
  button.addEventListener('click', () => {
    const closed = el.statusCard.classList.toggle('collapsed');
    button.textContent = closed ? 'Stats' : 'Hide stats';
  });
}

function setupModePanel() {
  const panel = document.createElement('div');
  panel.id = 'modePanel';
  panel.innerHTML = `
    <button id='modeToggle' type='button'>Hand · Natural</button>
    <div id='modeSheet' class='card'>
      <p class='controlLabel'>Tracking mode</p>
      <div class='buttonRow'><button class='modeChoice' data-mode='hand'>Hand</button><button class='modeChoice' data-mode='face'>Face</button><button class='modeChoice' data-mode='body'>Body</button></div>
      <p class='controlLabel'>Model style</p>
      <div class='buttonRow two'><button class='modeChoice' data-style='stick'>Stick</button><button class='modeChoice' data-style='natural'>Natural</button></div>
    </div>`;
  el.app.appendChild(panel);
  panel.querySelector('#modeToggle').onclick = () => panel.classList.toggle('open');
  panel.querySelectorAll('[data-mode]').forEach(b => b.onclick = () => switchMode(b.dataset.mode));
  panel.querySelectorAll('[data-style]').forEach(b => b.onclick = () => switchStyle(b.dataset.style));
  refreshButtons();
}

function refreshButtons() {
  document.querySelectorAll('[data-mode]').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));
  document.querySelectorAll('[data-style]').forEach(b => b.classList.toggle('active', b.dataset.style === style));
  const t = document.getElementById('modeToggle');
  if (t) t.textContent = label(mode) + ' · ' + label(style);
}
function label(x) { return x === 'hand' ? 'Hand' : x === 'face' ? 'Face' : x === 'body' ? 'Body' : x === 'stick' ? 'Stick' : 'Natural'; }
function updateLabels() { el.model.textContent = 'Model: ' + label(mode) + ' ' + label(style); el.tracker.textContent = 'Tracker: ' + (trackerMode || 'waiting'); }

async function switchMode(next) {
  mode = next;
  misses = 0; analyzed = 0; tracked = 0; triedRelaxed = false;
  rebuildModel(); refreshButtons(); updateLabels(); clearOverlay();
  if (running) await initTracker(false);
}
function switchStyle(next) { style = next; rebuildModel(); refreshButtons(); updateLabels(); }

async function startDemo() {
  el.startButton.disabled = true; el.startButton.textContent = 'Starting camera...'; el.startError.style.display = 'none';
  setStatus('Requesting camera...', 'warn');
  try {
    await startCamera();
    el.startButton.textContent = 'Loading libraries...'; setStatus('Loading libraries...', 'warn');
    await loadLibraries(); initScene(); rebuildModel();
    el.startButton.textContent = 'Loading tracker...'; await initTracker(false);
    running = true; el.startPanel.style.display = 'none'; setStatus('Tracking is running. Use the mode pill to switch.', 'warn');
  } catch (err) {
    console.error(err); stopCamera(); running = false; el.startButton.disabled = false; el.startButton.textContent = 'Try again';
    el.startError.textContent = friendlyError(err); el.startError.style.display = 'block'; setStatus('Could not start.', 'bad');
  }
}

async function startCamera() {
  if (!window.isSecureContext) throw new Error('Camera access needs HTTPS or localhost.');
  const tries = [
    { audio: false, video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 } } },
    { audio: false, video: { facingMode: 'user' } },
    { audio: false, video: true }
  ];
  let last;
  for (const opts of tries) {
    try {
      el.video.srcObject = await navigator.mediaDevices.getUserMedia(opts);
      el.video.setAttribute('playsinline', ''); el.video.muted = true;
      await waitForVideo(el.video); await el.video.play(); resizeOverlay(); return;
    } catch (err) { last = err; stopCamera(); }
  }
  throw last || new Error('No camera stream found.');
}
function waitForVideo(v) { return new Promise((res, rej) => { const t = setTimeout(() => rej(new Error('Camera opened, but video did not become ready.')), 9000); function ok(){ if(v.videoWidth){ clearTimeout(t); res(); } } v.onloadedmetadata = ok; v.oncanplay = ok; v.play().catch(()=>{}); }); }
function stopCamera() { if (el.video.srcObject) el.video.srcObject.getTracks().forEach(t => t.stop()); el.video.srcObject = null; }

async function loadLibraries() {
  const m = await Promise.all([import('three'), import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs')]);
  THREE = m[0]; FilesetResolver = m[1].FilesetResolver; HandLandmarker = m[1].HandLandmarker; FaceLandmarker = m[1].FaceLandmarker; PoseLandmarker = m[1].PoseLandmarker;
}
async function initTracker(relaxed) {
  loadingTracker = true; setStatus('Loading ' + label(mode).toLowerCase() + ' tracker...', 'warn');
  const fileset = await (vision || (vision = FilesetResolver.forVisionTasks(WASM)));
  try { if (tracker && tracker.close) tracker.close(); } catch (e) {}
  const delegate = relaxed ? 'CPU' : 'GPU';
  try {
    if (mode === 'hand') tracker = await HandLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: HAND_MODEL, delegate }, runningMode: 'VIDEO', numHands: 1, minHandDetectionConfidence: relaxed ? 0.12 : 0.25, minHandPresenceConfidence: relaxed ? 0.12 : 0.25, minTrackingConfidence: relaxed ? 0.12 : 0.25 });
    else if (mode === 'face') tracker = await FaceLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: FACE_MODEL, delegate }, runningMode: 'VIDEO', numFaces: 1, minFaceDetectionConfidence: relaxed ? 0.12 : 0.25, minFacePresenceConfidence: relaxed ? 0.12 : 0.25, minTrackingConfidence: relaxed ? 0.12 : 0.25 });
    else tracker = await PoseLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: POSE_MODEL, delegate }, runningMode: 'VIDEO', numPoses: 1, minPoseDetectionConfidence: relaxed ? 0.12 : 0.25, minPosePresenceConfidence: relaxed ? 0.12 : 0.25, minTrackingConfidence: relaxed ? 0.12 : 0.25 });
    trackerMode = label(mode) + ' ' + delegate;
  } catch (err) { if (!relaxed) return initTracker(true); throw err; }
  finally { loadingTracker = false; updateLabels(); }
}

function initScene() {
  if (scene) return;
  yAxis = new THREE.Vector3(0,1,0); tmp1 = new THREE.Vector3(); tmp2 = new THREE.Vector3();
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2)); renderer.setSize(innerWidth, innerHeight); renderer.outputColorSpace = THREE.SRGBColorSpace;
  el.fallback.style.display = 'none'; el.stage.appendChild(renderer.domElement);
  scene = new THREE.Scene(); camera3d = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.1, 100); camera3d.position.set(0, 0.05, 8.3);
  scene.add(new THREE.HemisphereLight(0xeaf7ff, 0x15182c, 3)); const key = new THREE.DirectionalLight(0xffffff, 2.4); key.position.set(2.8,4.4,4.8); scene.add(key);
  const rim = new THREE.DirectionalLight(0x76e3ff, 1.5); rim.position.set(-4.4,1,-2.4); scene.add(rim);
  grid = new THREE.Group(); scene.add(grid); makeGrid(); root = new THREE.Group(); scene.add(root); requestAnimationFrame(animate);
}

function rebuildModel() {
  if (!root || !THREE) return;
  while (root.children.length) root.remove(root.children[0]); meshes = { lines: [], joints: [], extras: [] };
  const count = mode === 'hand' ? 21 : mode === 'body' ? 33 : FACE_IDS.length;
  connections = mode === 'hand' ? HAND_CONN : mode === 'body' ? POSE_CONN : FACE_CONN;
  points = defaultPoints(count); target = points.map(p => p.clone()); smooth = points.map(p => p.clone());
  const lineMat = new THREE.MeshStandardMaterial({ color: style === 'stick' ? 0x90f4ff : 0xe9a47b, roughness: 0.5 });
  const jointMat = new THREE.MeshStandardMaterial({ color: style === 'stick' ? 0xffffff : 0xffd1b8, roughness: 0.45 });
  const cyl = new THREE.CylinderGeometry(1,1,1,18,1,false), sph = new THREE.SphereGeometry(1,20,12);
  connections.forEach(c => { const m = new THREE.Mesh(cyl, lineMat); m.userData = { a: c[0], b: c[1] }; root.add(m); meshes.lines.push(m); });
  points.forEach((p,i) => { const m = new THREE.Mesh(sph, jointMat); m.userData.index = i; root.add(m); meshes.joints.push(m); });
  if (style === 'natural') addNaturalExtras();
  updateLabels();
}
function defaultPoints(count) { const out = []; for (let i=0;i<count;i++) out.push(new THREE.Vector3((i%5-2)*0.22, (Math.floor(i/5)-2)*-0.22, 0)); return out; }
function addNaturalExtras() {
  const skin = new THREE.MeshStandardMaterial({ color: 0xf0b18c, roughness: 0.62, transparent: true, opacity: 0.78, side: THREE.DoubleSide });
  if (mode === 'hand') { const palm = new THREE.Mesh(new THREE.SphereGeometry(0.48,24,14), skin); palm.scale.set(1.0, 0.75, 0.22); root.add(palm); meshes.extras.push(palm); }
  if (mode === 'face') { const head = new THREE.Mesh(new THREE.SphereGeometry(0.9,32,20), skin); head.scale.set(0.85,1.08,0.55); root.add(head); meshes.extras.push(head); const eyeMat = new THREE.MeshStandardMaterial({ color: 0xffffff }); [-0.27,0.27].forEach(x => { const eye = new THREE.Mesh(new THREE.SphereGeometry(0.07,16,8), eyeMat); eye.position.set(x,0.13,0.43); root.add(eye); meshes.extras.push(eye); }); }
  if (mode === 'body') { const torso = new THREE.Mesh(new THREE.BoxGeometry(1.2,1.45,0.22), new THREE.MeshStandardMaterial({ color: 0x5bd7ff, roughness: 0.55, transparent: true, opacity: 0.82 })); torso.position.set(0,-0.35,-0.03); root.add(torso); meshes.extras.push(torso); const head = new THREE.Mesh(new THREE.SphereGeometry(0.28,24,16), skin); head.position.set(0,1.2,0); root.add(head); meshes.extras.push(head); }
}

function animate(now) {
  requestAnimationFrame(animate); if (!renderer) return;
  const dt = Math.max(1, now - lastFrame); fps = fps * 0.9 + (1000 / dt) * 0.1; lastFrame = now;
  if (running && tracker && !loadingTracker && el.video.readyState >= 2 && now - lastDetect > 32) { lastDetect = now; detect(now); }
  const follow = misses > 10 ? 0.06 : 0.42; for (let i=0;i<smooth.length;i++) smooth[i].lerp(target[i], follow);
  drawModel(); if (grid) grid.rotation.z = now * 0.000035; renderer.render(scene, camera3d);
  el.fps.textContent = 'FPS: ' + Math.round(fps); el.videoSize.textContent = 'Video: ' + (el.video.videoWidth || '?') + 'x' + (el.video.videoHeight || '?');
}
function detect(now) { analyzed++; el.analyzed.textContent = 'Analyzed: ' + analyzed; try { handle(tracker.detectForVideo(el.video, now)); } catch(e) { console.error(e); setStatus('Tracker error. Retrying...', 'bad'); } }
function handle(res) {
  clearOverlay(); let landmarks, world, score = 0.9, name = label(mode) + ': tracked';
  if (mode === 'hand') { if (!res.landmarks || !res.landmarks.length) return noDetection('No hand detected.'); landmarks = res.landmarks[0]; world = res.worldLandmarks && res.worldLandmarks[0]; const h = res.handednesses && res.handednesses[0] && res.handednesses[0][0]; if (h) { name = 'Hand: ' + h.categoryName; score = h.score; } target = convertPoints(landmarks, world, 21); drawOverlay(landmarks, HAND_CONN); }
  else if (mode === 'face') { if (!res.faceLandmarks || !res.faceLandmarks.length) return noDetection('No face detected.'); landmarks = res.faceLandmarks[0]; target = FACE_IDS.map(id => map2d(landmarks[id], 5.0, 5.6)); centerAndScale(target, [4,6,8]); drawFaceOverlay(landmarks); }
  else { if (!res.landmarks || !res.landmarks.length) return noDetection('No body detected. Step back.'); landmarks = res.landmarks[0]; world = res.worldLandmarks && res.worldLandmarks[0]; target = convertPoints(landmarks, world, 33); drawOverlay(landmarks, POSE_CONN); }
  gotDetection(name, score);
}
function noDetection(msg) { misses++; if (misses > 8) { setStatus(msg, 'bad'); el.noHand.textContent = msg; el.noHand.classList.add('show'); el.handedness.textContent = label(mode) + ': none'; el.confidence.textContent = 'Confidence: 0%'; } if (!triedRelaxed && analyzed > 100 && tracked === 0) { triedRelaxed = true; initTracker(true).catch(console.warn); } }
function gotDetection(name, score) { misses = 0; tracked++; el.tracked.textContent = 'Tracked: ' + tracked; el.noHand.classList.remove('show'); el.handedness.textContent = name; el.confidence.textContent = 'Confidence: ' + Math.round(score * 100) + '%'; setStatus(name.replace(':','') + ' detected.', 'good'); }
function convertPoints(lm, world, count) { let arr; if (world && world.length >= count) arr = world.slice(0,count).map(p => new THREE.Vector3(-p.x*4, -p.y*4, -p.z*4)); else arr = lm.slice(0,count).map(p => map2d(p, 5.2, 6.0)); centerAndScale(arr, mode === 'hand' ? [0,5,9,13,17] : [11,12,23,24]); return arr; }
function map2d(p, w, h) { return new THREE.Vector3((0.5 - p.x) * w, (0.5 - p.y) * h, -p.z * 4); }
function centerAndScale(arr, ids) { const c = new THREE.Vector3(); ids.forEach(i => arr[i] && c.add(arr[i])); c.multiplyScalar(1 / ids.length); arr.forEach(p => p.sub(c)); const d = ids.length > 2 && arr[ids[1]] && arr[ids[2]] ? dist(arr[ids[1]], arr[ids[2]]) : 1; const s = THREE.MathUtils.clamp(1.25 / Math.max(d, 0.1), 0.6, 2.4); arr.forEach(p => p.multiplyScalar(s)); }
function drawModel() { for (const m of meshes.lines) { if (smooth[m.userData.a] && smooth[m.userData.b]) placeCylinder(m, smooth[m.userData.a], smooth[m.userData.b], radius(m.userData.a, m.userData.b)); } for (const j of meshes.joints) { const p = smooth[j.userData.index]; if (p) { j.position.copy(p); j.scale.setScalar(style === 'stick' ? 0.045 : 0.085); } } }
function radius(a,b) { const base = style === 'stick' ? 0.018 : mode === 'body' ? 0.07 : 0.055; return base * ((mode === 'hand' && (TIP.has(a) || TIP.has(b))) ? 0.72 : 1); }
function placeCylinder(m, a, b, r) { const d = tmp1.subVectors(b,a), len = Math.max(d.length(), 0.0001); m.position.copy(tmp2.copy(a).add(b).multiplyScalar(0.5)); m.quaternion.setFromUnitVectors(yAxis, d.normalize()); m.scale.set(r, len, r); }
function drawOverlay(lm, conns) { const w=el.overlay.width,h=el.overlay.height; ctx.save(); ctx.lineCap='round'; conns.forEach(c => { const a=lm[c[0]], b=lm[c[1]]; if(!a||!b) return; ctx.beginPath(); ctx.moveTo(a.x*w,a.y*h); ctx.lineTo(b.x*w,b.y*h); ctx.lineWidth=5; ctx.strokeStyle='rgba(110,214,255,.92)'; ctx.stroke(); ctx.lineWidth=1.5; ctx.strokeStyle='rgba(0,0,0,.55)'; ctx.stroke(); }); lm.forEach((p,i)=>{ if(!p) return; ctx.beginPath(); ctx.arc(p.x*w,p.y*h,TIP.has(i)?7:4,0,Math.PI*2); ctx.fillStyle=TIP.has(i)?'#90f4ff':'#fff'; ctx.fill(); ctx.lineWidth=2; ctx.strokeStyle='rgba(0,0,0,.62)'; ctx.stroke(); }); ctx.restore(); }
function drawFaceOverlay(lm) { const w=el.overlay.width,h=el.overlay.height; ctx.save(); FACE_CONN.forEach(c=>{ const a=lm[FACE_IDS[c[0]]], b=lm[FACE_IDS[c[1]]]; if(!a||!b)return; ctx.beginPath(); ctx.moveTo(a.x*w,a.y*h); ctx.lineTo(b.x*w,b.y*h); ctx.lineWidth=4; ctx.strokeStyle='rgba(110,214,255,.92)'; ctx.stroke(); }); FACE_IDS.forEach(id=>{ const p=lm[id]; if(!p)return; ctx.beginPath(); ctx.arc(p.x*w,p.y*h,4.5,0,Math.PI*2); ctx.fillStyle='#fff'; ctx.fill(); ctx.strokeStyle='rgba(0,0,0,.62)'; ctx.stroke(); }); ctx.restore(); }
function clearOverlay() { resizeOverlay(); ctx.clearRect(0,0,el.overlay.width,el.overlay.height); }
function makeGrid() { const lm=new THREE.LineBasicMaterial({color:0x66d9ff,transparent:true,opacity:.09}), rm=new THREE.LineBasicMaterial({color:0xffffff,transparent:true,opacity:.055}); for(let r=1.2;r<=6;r+=.8){const p=[];for(let i=0;i<=128;i++){const a=i/128*Math.PI*2;p.push(new THREE.Vector3(Math.cos(a)*r,Math.sin(a)*r,-3.2));}grid.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p),rm));} }
function resizeOverlay(){ const r=el.overlay.getBoundingClientRect(), d=Math.min(devicePixelRatio||1,2), w=Math.max(1,Math.round(r.width*d)), h=Math.max(1,Math.round(r.height*d)); if(el.overlay.width!==w||el.overlay.height!==h){el.overlay.width=w;el.overlay.height=h;ctx=el.overlay.getContext('2d',{alpha:true});} }
function resizeAll(){ resizeOverlay(); if(!renderer)return; renderer.setPixelRatio(Math.min(devicePixelRatio||1,2)); renderer.setSize(innerWidth,innerHeight); camera3d.aspect=innerWidth/innerHeight; camera3d.updateProjectionMatrix(); }
function setStatus(text, state){ el.statusText.textContent=text; el.stateDot.classList.toggle('good',state==='good'); el.stateDot.classList.toggle('bad',state==='bad'); }
function showLog(text){ el.startLog.textContent=text; el.startLog.style.display='block'; }
function friendlyError(e){ const d='\n\nTechnical details: '+(e.name||'Error')+': '+(e.message||String(e)); if(!isSecureContext)return 'Camera access needs HTTPS or localhost.'+d; if(/permission|denied|NotAllowed/i.test(e.name+' '+e.message))return 'Chrome denied camera access. Allow camera for this site and try again.'+d; return 'Something failed while starting the camera, tracker, or 3D view.'+d; }
function dist(a,b){ return Math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2); }
