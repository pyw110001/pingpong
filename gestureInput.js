import { FilesetResolver, GestureRecognizer, PoseLandmarker } from '@mediapipe/tasks-vision';

const WASM_BASE_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const POSE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task';
const GESTURE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task';

const RIGHT_SHOULDER = 12;
const RIGHT_WRIST = 16;
const MIN_VISIBILITY = 0.45;
const CONFIRM_HOLD_MS = 350;
const CONFIRM_COOLDOWN_MS = 1000;
const FIST_MIN_SCORE = 0.62;
const PREVIEW_LANDMARKS = new Set([0, 11, 12, 13, 14, 15, 16, 23, 24]);

async function createPoseLandmarker(fileset, delegate) {
  return PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: POSE_MODEL_URL,
      delegate
    },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
}

async function createGestureRecognizer(fileset, delegate) {
  return GestureRecognizer.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: GESTURE_MODEL_URL,
      delegate
    },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    cannedGesturesClassifierOptions: {
      scoreThreshold: 0.5,
      categoryAllowlist: ['Closed_Fist']
    }
  });
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function landmarkConfidence(landmark) {
  return Math.min(landmark?.visibility ?? 0, landmark?.presence ?? 1);
}

function statusDetail(code, message) {
  return { code, message };
}

function mirroredPoint(landmark, width, height) {
  return {
    x: (1 - clamp(landmark.x, 0, 1)) * width,
    y: clamp(landmark.y, 0, 1) * height
  };
}

export function createGestureInput({ onPoint, onConfirm, onFist, onStatus, previewContainer } = {}) {
  let disposed = false;
  let stream = null;
  let video = null;
  let previewCanvas = null;
  let previewCtx = null;
  let landmarker = null;
  let gestureRecognizer = null;
  let rafId = 0;
  let smoothX = 0;
  let smoothY = 0;
  let hasSmoothPoint = false;
  let noPersonSince = 0;
  let confirmStartedAt = 0;
  let lastConfirmAt = 0;
  let fistActive = false;
  let lastVideoTime = -1;

  const setStatus = (code, message) => {
    onStatus?.(statusDetail(code, message));
  };

  const setupPreview = () => {
    if (!previewContainer || !video) return;

    previewContainer.textContent = '';
    video.style.cssText = `
      position:absolute; inset:0; width:100%; height:100%; object-fit:cover;
      transform:scaleX(-1); pointer-events:none;
    `;
    video.setAttribute('aria-hidden', 'true');
    previewContainer.appendChild(video);

    previewCanvas = document.createElement('canvas');
    previewCanvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;';
    previewCanvas.setAttribute('aria-hidden', 'true');
    previewContainer.appendChild(previewCanvas);
    previewCtx = previewCanvas.getContext('2d');
  };

  const resizePreviewCanvas = () => {
    if (!previewCanvas || !previewContainer) return null;

    const rect = previewContainer.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));

    if (previewCanvas.width !== width || previewCanvas.height !== height) {
      previewCanvas.width = width;
      previewCanvas.height = height;
    }

    return { width, height };
  };

  const clearPreview = () => {
    const size = resizePreviewCanvas();
    if (!previewCtx || !size) return;
    previewCtx.clearRect(0, 0, size.width, size.height);
  };

  const drawPreview = (pose, wrist, shoulder, handOverShoulder) => {
    const size = resizePreviewCanvas();
    if (!previewCtx || !size) return;

    const { width, height } = size;
    previewCtx.clearRect(0, 0, width, height);

    if (!pose) {
      previewCtx.strokeStyle = 'rgba(255,255,255,0.18)';
      previewCtx.lineWidth = 1;
      previewCtx.setLineDash([4, 6]);
      previewCtx.strokeRect(10, 10, width - 20, height - 20);
      previewCtx.setLineDash([]);
      return;
    }

    previewCtx.lineCap = 'round';
    previewCtx.lineJoin = 'round';

    previewCtx.strokeStyle = 'rgba(87,255,183,0.82)';
    previewCtx.lineWidth = Math.max(2, width * 0.012);
    for (const connection of PoseLandmarker.POSE_CONNECTIONS) {
      const from = pose[connection.start];
      const to = pose[connection.end];
      if (!from || !to) continue;
      if (landmarkConfidence(from) < 0.35 || landmarkConfidence(to) < 0.35) continue;
      if (!PREVIEW_LANDMARKS.has(connection.start) && !PREVIEW_LANDMARKS.has(connection.end)) continue;

      const a = mirroredPoint(from, width, height);
      const b = mirroredPoint(to, width, height);
      previewCtx.beginPath();
      previewCtx.moveTo(a.x, a.y);
      previewCtx.lineTo(b.x, b.y);
      previewCtx.stroke();
    }

    if (shoulder && landmarkConfidence(shoulder) >= 0.35) {
      const p = mirroredPoint(shoulder, width, height);
      previewCtx.fillStyle = 'rgba(120,190,255,0.92)';
      previewCtx.beginPath();
      previewCtx.arc(p.x, p.y, Math.max(4, width * 0.018), 0, Math.PI * 2);
      previewCtx.fill();
    }

    if (wrist) {
      const p = mirroredPoint(wrist, width, height);
      const radius = Math.max(8, width * 0.04);
      previewCtx.fillStyle = handOverShoulder ? 'rgba(255,224,112,0.95)' : 'rgba(87,255,183,0.96)';
      previewCtx.strokeStyle = 'rgba(0,0,0,0.55)';
      previewCtx.lineWidth = Math.max(2, width * 0.01);
      previewCtx.beginPath();
      previewCtx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      previewCtx.fill();
      previewCtx.stroke();

      previewCtx.strokeStyle = handOverShoulder ? 'rgba(255,224,112,0.5)' : 'rgba(87,255,183,0.42)';
      previewCtx.lineWidth = Math.max(2, width * 0.008);
      previewCtx.beginPath();
      previewCtx.arc(p.x, p.y, radius * 1.8, 0, Math.PI * 2);
      previewCtx.stroke();
    }
  };

  const stop = () => {
    disposed = true;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    if (video) {
      video.remove();
      video = null;
    }
    previewCanvas?.remove();
    previewCanvas = null;
    previewCtx = null;
    landmarker?.close?.();
    landmarker = null;
    gestureRecognizer?.close?.();
    gestureRecognizer = null;
  };

  const emitNoPerson = (now) => {
    if (!noPersonSince) noPersonSince = now;
    if (now - noPersonSince > 700) {
      setStatus('no-person', '手势：请露出右手');
    }
    confirmStartedAt = 0;
  };

  const processPose = (result, now) => {
    const pose = result?.landmarks?.[0];
    const wrist = pose?.[RIGHT_WRIST];
    const shoulder = pose?.[RIGHT_SHOULDER];
    const confidence = landmarkConfidence(wrist);
    const shoulderConfidence = landmarkConfidence(shoulder);
    const handOverShoulder = shoulder && shoulderConfidence >= MIN_VISIBILITY && wrist && wrist.y < shoulder.y - 0.05;

    if (!wrist || confidence < MIN_VISIBILITY) {
      drawPreview(pose, wrist, shoulder, false);
      emitNoPerson(now);
      return;
    }

    drawPreview(pose, wrist, shoulder, handOverShoulder);

    noPersonSince = 0;

    // Front-camera kiosk controls should feel mirrored: raise/move your right
    // hand to the right, and the paddle moves to the right on screen.
    const rawX = 1 - clamp(wrist.x, 0, 1);
    const rawY = clamp(wrist.y, 0, 1);
    const nextX = rawX * 2 - 1;
    const nextY = rawY * 2 - 1;

    if (!hasSmoothPoint) {
      smoothX = nextX;
      smoothY = nextY;
      hasSmoothPoint = true;
    } else {
      smoothX += (nextX - smoothX) * 0.38;
      smoothY += (nextY - smoothY) * 0.38;
    }

    onPoint?.({
      x: clamp(smoothX, -1, 1),
      y: clamp(smoothY, -1, 1),
      confidence,
      source: 'right_wrist'
    });
    setStatus('ready', `手势：右手腕 ${Math.round(confidence * 100)}%`);

    if (!handOverShoulder) {
      confirmStartedAt = 0;
      return;
    }

    if (!confirmStartedAt) confirmStartedAt = now;
    if (now - confirmStartedAt >= CONFIRM_HOLD_MS && now - lastConfirmAt >= CONFIRM_COOLDOWN_MS) {
      lastConfirmAt = now;
      confirmStartedAt = 0;
      onConfirm?.();
      setStatus('confirmed', '手势：已确认');
    }
  };

  const processHandGesture = (result) => {
    const gesture = result?.gestures?.[0]?.[0];
    const isFist = gesture?.categoryName === 'Closed_Fist' && (gesture.score ?? 0) >= FIST_MIN_SCORE;

    if (!isFist) {
      fistActive = false;
      return;
    }

    if (fistActive) return;
    fistActive = true;
    onFist?.({
      confidence: gesture.score ?? 0,
      source: 'closed_fist'
    });
    setStatus('confirmed', `手势：握拳 ${Math.round((gesture.score ?? 0) * 100)}%`);
  };

  const predict = () => {
    if (disposed || !video || !landmarker || !gestureRecognizer) return;

    const now = performance.now();
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      try {
        processPose(landmarker.detectForVideo(video, now), now);
        processHandGesture(gestureRecognizer.recognizeForVideo(video, now));
      } catch (error) {
        setStatus('error', '手势：识别错误');
      }
    }

    rafId = requestAnimationFrame(predict);
  };

  const start = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus('no-camera', '手势：摄像头不可用');
      return;
    }

    setStatus('loading', '手势：正在打开摄像头');

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30, max: 30 }
        },
        audio: false
      });

      video = document.createElement('video');
      video.muted = true;
      video.playsInline = true;
      video.autoplay = true;
      video.srcObject = stream;
      if (previewContainer) {
        setupPreview();
      } else {
        video.style.cssText = 'position:fixed;width:1px;height:1px;opacity:0;pointer-events:none;left:-10px;top:-10px;';
        video.setAttribute('aria-hidden', 'true');
        document.body.appendChild(video);
      }
      await video.play();

      setStatus('loading', '手势：正在加载识别模型');
      const fileset = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
      try {
        landmarker = await createPoseLandmarker(fileset, 'GPU');
        gestureRecognizer = await createGestureRecognizer(fileset, 'GPU');
      } catch (error) {
        landmarker = await createPoseLandmarker(fileset, 'CPU');
        gestureRecognizer = await createGestureRecognizer(fileset, 'CPU');
      }

      if (disposed) {
        stop();
        return;
      }

      setStatus('ready', '手势：已就绪');
      predict();
    } catch (error) {
      const denied = error?.name === 'NotAllowedError' || error?.name === 'PermissionDeniedError';
      setStatus(denied ? 'permission-denied' : 'no-camera', denied ? '手势：摄像头权限被阻止' : '手势：摄像头不可用');
      stop();
    }
  };

  void start();

  return { stop };
}
