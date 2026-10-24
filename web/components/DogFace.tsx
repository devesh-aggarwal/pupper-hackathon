import { robotBaseUrl } from '../constants';
import React, { useEffect, useState, useRef } from 'react';
import { SystemStatus, AppConfig, ScreenDisplay } from '../types';
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

interface DogFaceProps {
  isListening: boolean;
  isSpeaking: boolean;
  audioLevel: number;
  systemStatus: SystemStatus | null;
  config: AppConfig;
  videoElement: HTMLVideoElement | null;
  display?: ScreenDisplay | null; // hackathon: big words on the screen (show_text, reminders, sound alerts)
  hearing?: string | null; // hackathon: what Pupper hears right now ("Speech", "Music"); null = silence
  caption?: string | null; // hackathon: what Pupper is saying, as text
  onDismissDisplay?: () => void; // hackathon: a tap on a screen card clears it
}

/** Black text on light colors (yellow, white), white on the rest. */
const textColorFor = (hex: string): string => {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex.trim());
  if (!m) return '#ffffff';
  const [r, g, b] = [m[1], m[2], m[3]].map(v => parseInt(v, 16));
  return (0.299 * r + 0.587 * g + 0.114 * b) > 170 ? '#111111' : '#ffffff';
};

// Interfaces for ZMQ Detections
interface PersonLocation {
    x: number;
    y: number;
    width: number;
    height: number;
    heading: number; // degrees
    elevation: number; // degrees
    id: number;
}

interface PeopleDetections {
    people: PersonLocation[];
    timestamp: number;
}

// Configuration
const EYE_TRACKING = {
  sensitivity: 0.6, 
  maxPupilOffset: 40.0, 
  maxEyeOffset: 60.0,
  combinedPupilRatio: 0.6,
  smoothing: 0.15,
};

// SVG Geometry helpers
const getQuadraticBezier = (start: {x:number, y:number}, ctrl: {x:number, y:number}, end: {x:number, y:number}) => {
  return `M ${start.x} ${start.y} Q ${ctrl.x} ${ctrl.y} ${end.x} ${end.y}`;
};

// Expressions. Every number blends smoothly toward the current emotion's preset.
//   lidTop: 0 open .. 1 closed; browY: raise in px; browTilt: degrees, + = inner ends
//   down (focused), - = worried
type Emotion = 'neutral' | 'happy' | 'listening' | 'sleepy' | 'surprised' | 'sad' | 'excited';
interface Expr { lidTop: number; browY: number; browTilt: number; }
const EXPRESSIONS: Record<Emotion, Expr> = {
  neutral:   { lidTop: 0.00, browY: 0,  browTilt: 0 },
  happy:     { lidTop: 0.00, browY: 10, browTilt: -3 },
  listening: { lidTop: 0.00, browY: 14, browTilt: -5 },
  sleepy:    { lidTop: 0.00, browY: -6, browTilt: 0 },
  surprised: { lidTop: 0.00, browY: 30, browTilt: 0 },
  sad:       { lidTop: 0.00, browY: 6,  browTilt: -16 },
  excited:   { lidTop: 0.00, browY: 18, browTilt: -4 },
};
const EXPR_KEYS = Object.keys(EXPRESSIONS.neutral) as (keyof Expr)[];

const Eye: React.FC<{
  id: string;
  cx: number;
  cy: number;
  scale: number;
  pupilOffset: { x: number; y: number };
  lid: number; // 0 open .. 1 shut: blinking, and half shut when sleepy
}> = ({ id, cx, cy, scale, pupilOffset, lid }) => {
  const colors = {
    ringOuter: '#2a2f36',
    ringBlue: '#4b95b5',
    underHighlight: '#326688',
    pupil: '#000000',
    gloss: 'rgba(255, 255, 255, 0.95)',
  };

  const rOuter = 140 * scale;
  const rBezel = 132 * scale;
  const rPupil = 100 * scale;

  const px = cx + pupilOffset.x;
  const py = cy + pupilOffset.y;

  // The lid is a black shape (the background) sliding down over the eye, clipped to it.
  const w = rOuter + 10 * scale;
  const lidEdge = cy - w + 2 * w * lid;

  return (
    <g>
      <defs>
        <clipPath id={`clip-${id}`}><circle cx={cx} cy={cy} r={rOuter + 6 * scale} /></clipPath>
      </defs>
      <g clipPath={`url(#clip-${id})`}>
        {/* Outer Dark Ring */}
        <circle cx={cx} cy={cy} r={rOuter} fill="none" stroke={colors.ringOuter} strokeWidth={8 * scale} />

        {/* Blue Bezel */}
        <circle cx={cx} cy={cy} r={rBezel} fill={colors.ringBlue} />

        {/* Pupil */}
        <circle cx={px} cy={py} r={rPupil} fill={colors.pupil} />

        {/* Under Highlight Arc */}
        <path
          d={`M ${px + (56*scale)} ${py + (56*scale)} A ${80*scale} ${80*scale} 0 0 1 ${px - (56*scale)} ${py + (56*scale)}`}
          fill="none"
          stroke={colors.underHighlight}
          strokeWidth={14 * scale}
          strokeLinecap="round"
        />

        {/* Little Blue Dot */}
        <circle cx={px + (76*scale)} cy={py + (34*scale)} r={8*scale} fill={colors.underHighlight} />

        {/* Gloss Highlights */}
        <circle cx={px - (42*scale)} cy={py - (54*scale)} r={26*scale} fill={colors.gloss} />
        <circle cx={px - (70*scale)} cy={py - (8*scale)} r={12*scale} fill={colors.gloss} />

        {/* Eyelid */}
        {lid > 0.01 && (
          <path d={`M ${cx - w} ${cy - w} L ${cx + w} ${cy - w} L ${cx + w} ${lidEdge} Q ${cx} ${lidEdge + 30 * scale} ${cx - w} ${lidEdge} Z`} fill="#000" />
        )}
      </g>
    </g>
  );
};

const Eyebrow: React.FC<{ cx: number; cy: number; scale: number; expr: Expr; side: 1 | -1 }> = ({ cx, cy, scale, expr, side }) => {
  // side: -1 left eye, 1 right eye. The inner end is toward the middle of the face.
  const dx = 88 * scale;
  const baseY = cy - (150 + expr.browY) * scale;
  const tilt = expr.browTilt * scale;
  const innerX = cx - side * dx, outerX = cx + side * dx;
  return (
    <path
      d={`M ${outerX} ${baseY - tilt * 0.3} Q ${cx} ${baseY - 45 * scale} ${innerX} ${baseY + tilt}`}
      fill="none"
      stroke="#3a3e46"
      strokeWidth={14 * scale}
      strokeLinecap="round"
    />
  );
};

const Nose: React.FC<{ cx: number; cy: number; scale: number }> = ({ cx, cy, scale }) => (
  <g>
    <defs>
      <radialGradient id="nose-grad" cx="45%" cy="35%" r="70%">
        <stop offset="0%" stopColor="#4b5563" />
        <stop offset="100%" stopColor="#111418" />
      </radialGradient>
    </defs>
    <path d={`M ${cx - 38 * scale} ${cy - 14 * scale}
              Q ${cx} ${cy - 30 * scale} ${cx + 38 * scale} ${cy - 14 * scale}
              Q ${cx + 30 * scale} ${cy + 26 * scale} ${cx} ${cy + 30 * scale}
              Q ${cx - 30 * scale} ${cy + 26 * scale} ${cx - 38 * scale} ${cy - 14 * scale} Z`}
          fill="url(#nose-grad)" />
    <ellipse cx={cx - 12 * scale} cy={cy - 12 * scale} rx={11 * scale} ry={5 * scale} fill="rgba(255,255,255,0.45)" />
  </g>
);

// The original mouth: a thin closed smile, and a cyan curve that opens with the voice.
const Mouth: React.FC<{ cx: number; y: number; scale: number; audioLevel: number; isSpeaking: boolean }> = ({ cx, y, scale, audioLevel, isSpeaking }) => (
  <>
    <path
      d={`M ${cx - 30 * scale} ${y} Q ${cx} ${y + 10 * scale} ${cx + 30 * scale} ${y}`}
      fill="none" stroke="#33363c" strokeWidth={4 * scale} strokeLinecap="round"
      style={{ opacity: isSpeaking ? 0 : 1, transition: 'opacity 0.1s' }}
    />
    <path
      d={`M ${cx - 35 * scale} ${y} Q ${cx} ${y + audioLevel * 120 * scale + 10 * scale} ${cx + 35 * scale} ${y}`}
      fill="none" stroke="#22d3ee" strokeWidth={5 * scale} strokeLinecap="round"
      style={{ opacity: isSpeaking ? 0.9 : 0, transition: 'opacity 0.05s' }}
    />
  </>
);

const DogFace: React.FC<DogFaceProps> = ({ isListening, isSpeaking, audioLevel, systemStatus, config, videoElement, display, caption, hearing, onDismissDisplay }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  
  // Layout State
  const [windowCenter, setWindowCenter] = useState({ x: 0, y: 0 });
  const [globalScale, setGlobalScale] = useState(1.0);
  
  // Tracking State
  const targetPosRef = useRef({ x: 0.5, y: 0.5 }); // Normalized 0..1
  const currentPosRef = useRef({ x: 0.5, y: 0.5 }); // Normalized 0..1
  const [renderPos, setRenderPos] = useState({ x: 0.5, y: 0.5 });
  
  // Detection State
  const lastDetectionTime = useRef<number>(0);

  // Critical Error Logic: Only error if we HAVE status and it's bad.
  // Don't error on null (connecting).
  const isConnected = !!systemStatus;
  const isLowBattery = systemStatus?.battery?.percentage != null && systemStatus.battery.percentage < 10;
  const isRobotInactive = systemStatus?.services?.robot === 'inactive';
  
  const hasCriticalError = isConnected && (isLowBattery || isRobotInactive);

  // Resize Handler
  useEffect(() => {
    const handleResize = () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      setWindowCenter({ x: w / 2, y: h / 2 });
      
      const wScale = w / 600; 
      const hScale = h / 500;
      // 0.85: a little room at the edges, so the eyes (and their tracking) aren't cut off.
      let newScale = Math.min(wScale, hScale) * 0.85;
      newScale = Math.max(0.35, Math.min(1.0, newScale));
      setGlobalScale(newScale);
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // --- SOURCE 1: SERVER (ZMQ) ---
  useEffect(() => {
    if (config.eyeTrackingSource !== 'server') return;
    
    let ws: WebSocket;
    let reconnectTimer: any;
    let isUnmounted = false;

    const connect = () => {
        if (isUnmounted) return;
        try {
            const httpUrl = robotBaseUrl(config.robotApiUrl);
            const wsUrl = httpUrl.replace(/^http/, 'ws') + '/ws/detections';
            
            ws = new WebSocket(wsUrl);
            ws.onopen = () => console.log("[Face] WS Connected");
            ws.onmessage = (event) => {
                try {
                    const data: PeopleDetections = JSON.parse(event.data);
                    
                    if (data.people && Array.isArray(data.people) && data.people.length > 0) {
                         // Find the person with the largest bounding box area
                         let largest: PersonLocation | null = null;
                         let maxArea = -1;
            
                         for (const p of data.people) {
                             const area = p.width * p.height;
                             if (area > maxArea) {
                                 maxArea = area;
                                 largest = p;
                             }
                         }
            
                         if (largest) {
                             // Calculate offset based on heading and elevation
                             // Based on Rust reference: offset = degrees / 90 * (window_size / 2)
                             // Mapping this to normalized coordinates [0, 1] centered at 0.5:
                             // offset_norm = (degrees / 90) * 0.5
                             
                             const h = Math.max(-90, Math.min(90, largest.heading));
                             const e = Math.max(-90, Math.min(90, largest.elevation));
            
                             const tx = 0.5 - (h / 90.0) * 0.5 * 3;
                             const ty = 1.0 - (e / 90.0) * 0.5 * 10;
            
                             targetPosRef.current = { x: tx, y: ty };
                             lastDetectionTime.current = Date.now();
                         }
                    } else if ((data as any).center) {
                        // Fallback for older format
                        const fallback = data as any;
                        targetPosRef.current = { x: fallback.center.x, y: fallback.center.y };
                        lastDetectionTime.current = Date.now();
                    }
                } catch(e) {}
            };
            ws.onclose = () => {
                if (!isUnmounted) reconnectTimer = setTimeout(connect, 2000);
            };
            ws.onerror = () => ws.close();
        } catch(e) {
            if (!isUnmounted) reconnectTimer = setTimeout(connect, 5000);
        }
    };

    connect();
    return () => {
        isUnmounted = true;
        if (ws) ws.close();
        clearTimeout(reconnectTimer);
    };
  }, [config.eyeTrackingSource, config.robotApiUrl]);

  // --- SOURCE 2: MEDIAPIPE ---
  useEffect(() => {
    if (config.eyeTrackingSource !== 'mediapipe' || !videoElement) return;

    let faceLandmarker: FaceLandmarker | null = null;
    let animationId: number;
    let lastVideoTime = -1;

    const setupMediaPipe = async () => {
        try {
            const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
            faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                numFaces: 1
            });
            predictLoop();
        } catch (e) {
            console.error("Failed to load MediaPipe:", e);
        }
    };

    const predictLoop = () => {
        if (faceLandmarker && videoElement && videoElement.currentTime !== lastVideoTime) {
            lastVideoTime = videoElement.currentTime;
            try {
                const results = faceLandmarker.detectForVideo(videoElement, performance.now());
                if (results.faceLandmarks && results.faceLandmarks.length > 0) {
                    // Get bounding box roughly from landmarks
                    // Simple average of points
                    let xSum = 0, ySum = 0;
                    const points = results.faceLandmarks[0];
                    for (const p of points) {
                        xSum += p.x;
                        ySum += p.y;
                    }
                    const avgX = xSum / points.length;
                    const avgY = ySum / points.length;
                    
                    // Mirror X because it's a selfie camera usually
                    targetPosRef.current = { x: 1.0 - avgX, y: avgY };
                    lastDetectionTime.current = Date.now();
                }
            } catch(e) {}
        }
        animationId = requestAnimationFrame(predictLoop);
    };

    setupMediaPipe();

    return () => {
        cancelAnimationFrame(animationId);
        if (faceLandmarker) faceLandmarker.close();
    };
  }, [config.eyeTrackingSource, videoElement]);

  // --- SOURCE 3: MOUSE (Always active as fallback if others timeout) ---
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      // If Explicitly 'mouse', always use it.
      // If 'server' or 'mediapipe', only use fallback if timeout > 1s
      const useMouse = config.eyeTrackingSource === 'mouse' || (Date.now() - lastDetectionTime.current > 1000);
      
      if (useMouse) {
        const nx = e.clientX / window.innerWidth;
        const ny = e.clientY / window.innerHeight;
        targetPosRef.current = { x: nx, y: ny };
      }
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [config.eyeTrackingSource]);

  // Emotion: picked from what is happening, highest priority first
  const emotion: Emotion =
    hasCriticalError ? 'sad'
    : display?.kind === 'sound' ? 'surprised'
    : display && (display.kind === 'found' || display.kind === 'goto') ? 'excited'
    : !isListening ? 'sleepy'
    : isSpeaking ? 'happy'
    : hearing ? 'listening'
    : 'neutral';
  const emotionRef = useRef<Emotion>(emotion);
  emotionRef.current = emotion;
  const exprRef = useRef<Expr>({ ...EXPRESSIONS.neutral });
  const [expr, setExpr] = useState<Expr>(exprRef.current);
  const [blink, setBlink] = useState(0);
  const [anim, setAnim] = useState({ breathe: 0, wiggle: 0 });

  // Blinking: every 2-6 s, sometimes a double blink. Sleepy blinks slowly.
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let raf = 0;
    const doBlink = (then: () => void) => {
      const dur = emotionRef.current === 'sleepy' ? 420 : 160;
      const t0 = performance.now();
      const step = (t: number) => {
        const k = Math.min(1, (t - t0) / dur);
        setBlink(Math.sin(k * Math.PI));
        if (k < 1) raf = requestAnimationFrame(step); else { setBlink(0); then(); }
      };
      raf = requestAnimationFrame(step);
    };
    const schedule = () => {
      timer = setTimeout(() => doBlink(() => {
        if (Math.random() < 0.2) timer = setTimeout(() => doBlink(schedule), 120);
        else schedule();
      }), 2000 + Math.random() * 4000);
    };
    schedule();
    return () => { clearTimeout(timer); cancelAnimationFrame(raf); };
  }, []);

  // Animation Loop for Smoothing
  useEffect(() => {
    let animFrame: number;
    
    const loop = () => {
      // Blend the expression toward the current emotion, so faces melt into each other
      const goal = EXPRESSIONS[emotionRef.current];
      const e = exprRef.current;
      for (const k of EXPR_KEYS) e[k] += (goal[k] - e[k]) * 0.08;
      setExpr({ ...e });
      const t = performance.now() / 1000;
      const mood = emotionRef.current;
      setAnim({
        breathe: Math.sin(t * (mood === 'sleepy' ? 0.9 : 1.6)),
        // drives the excited bounce
        wiggle: (mood === 'excited' ? 9 : mood === 'happy' ? 4 : 1) * Math.sin(t * (mood === 'excited' ? 9 : 3)),
      });
      const target = targetPosRef.current;
      const current = currentPosRef.current;
      
      const dx = target.x - current.x;
      const dy = target.y - current.y;
      const smooth = EYE_TRACKING.smoothing;
      
      currentPosRef.current = {
          x: current.x + dx * smooth,
          y: current.y + dy * smooth
      };

      setRenderPos({ ...currentPosRef.current });
      animFrame = requestAnimationFrame(loop);
    };
    
    loop();
    return () => cancelAnimationFrame(animFrame);
  }, []);

  // Calculate Eye Components
  const calculateEyeOffset = () => {
    const rx = (renderPos.x - 0.5) * 2.0; 
    const ry = (renderPos.y - 0.5) * 2.0;

    const sx = rx * EYE_TRACKING.sensitivity;
    const sy = ry * EYE_TRACKING.sensitivity;

    const maxEye = EYE_TRACKING.maxEyeOffset * globalScale;
    const ex = sx * maxEye;
    const ey = sy * maxEye;

    const maxPupil = EYE_TRACKING.maxPupilOffset * globalScale;
    let px = sx * (maxPupil / 0.5); 
    let py = sy * (maxPupil / 0.5);

    const pLen = Math.sqrt(px*px + py*py);
    if (pLen > maxPupil) {
        px = (px / pLen) * maxPupil;
        py = (py / pLen) * maxPupil;
    }

    return {
      eye: { x: ex, y: ey },
      pupil: { x: px, y: py }
    };
  };

  const offsets = calculateEyeOffset();
  
  // Layout Constants
  const eyeSeparation = 180 * globalScale;
  const yOffset = -80 * globalScale + anim.breathe * 4 * globalScale; // eyes in upper half, gently breathing
  const lid = Math.min(1, expr.lidTop + blink * (1 - expr.lidTop));
  const bounce = emotion === 'excited' ? Math.abs(anim.wiggle) * 1.2 * globalScale : 0;

  const leftEyeCenter = { x: windowCenter.x - eyeSeparation + offsets.eye.x, y: windowCenter.y + yOffset + offsets.eye.y - bounce };
  const rightEyeCenter = { x: windowCenter.x + eyeSeparation + offsets.eye.x, y: windowCenter.y + yOffset + offsets.eye.y - bounce };
  const nosePos = { x: windowCenter.x + offsets.eye.x * 0.5, y: windowCenter.y + yOffset + 180 * globalScale + offsets.eye.y * 0.5 - bounce };
  const mouthBaseY = nosePos.y + 80 * globalScale;

  return (
    <div className="w-full h-full bg-black relative overflow-hidden flex items-center justify-center">
      {/* Background Ambience */}
      <div className={`absolute inset-0 bg-blue-900/5 transition-opacity duration-500 ${isListening ? 'opacity-100' : 'opacity-0'}`} />

      <svg 
        ref={svgRef}
        className="w-full h-full"
        viewBox={`0 0 ${window.innerWidth} ${window.innerHeight}`}
      >

        <Eyebrow cx={leftEyeCenter.x} cy={leftEyeCenter.y} scale={globalScale} expr={expr} side={-1} />
        <Eye id="l" cx={leftEyeCenter.x} cy={leftEyeCenter.y} scale={globalScale} pupilOffset={offsets.pupil} lid={lid} />
        <Eyebrow cx={rightEyeCenter.x} cy={rightEyeCenter.y} scale={globalScale} expr={expr} side={1} />
        <Eye id="r" cx={rightEyeCenter.x} cy={rightEyeCenter.y} scale={globalScale} pupilOffset={offsets.pupil} lid={lid} />

        <Nose cx={nosePos.x} cy={nosePos.y} scale={globalScale} />
        <Mouth cx={nosePos.x} y={mouthBaseY} scale={globalScale} audioLevel={audioLevel} isSpeaking={isSpeaking} />

      </svg>

      {/* Hackathon: captions, so a person who can't hear Pupper can read it */}
      {caption && display?.kind !== 'sound' && (
        <div className="absolute bottom-20 left-0 right-0 flex justify-center px-6 z-30 pointer-events-none">
          <div className="bg-black/70 text-white rounded-2xl px-6 py-3 max-w-[90%] text-center font-semibold leading-snug"
               style={{ fontSize: `${Math.max(18, 30 * globalScale)}px` }}>
            {caption}
          </div>
        </div>
      )}

      {/* Hackathon: what Pupper hears right now, small, in a corner (nothing for silence) */}
      {hearing && display?.kind !== 'sound' && (
        <div className="absolute top-4 left-4 z-30 pointer-events-none bg-black/60 text-white rounded-xl px-4 py-2 font-semibold"
             style={{ fontSize: `${Math.max(14, 22 * globalScale)}px` }}>
          {'\u{1F442} '}{hearing}
        </div>
      )}

      {/* Hackathon: a sound alert takes over the whole screen (urgent ones in red) */}
      {display && display.kind === 'sound' && (
        <div key={`${display.text}-${display.detail}`}
             className="alert-in absolute inset-0 z-40 flex flex-col items-center justify-center text-center px-6"
             style={{ background: display.color, color: textColorFor(display.color) }}>
          <div className="font-black uppercase tracking-wide leading-none break-words max-w-full"
               style={{ fontSize: `${Math.max(40, (display.text.length > 14 ? 72 : 110) * globalScale)}px` }}>
            {display.text}
          </div>
          {display.detail && (
            <div className="mt-6 font-semibold opacity-90 break-words max-w-full" style={{ fontSize: `${Math.max(18, 34 * globalScale)}px` }}>
              {display.detail}
            </div>
          )}
        </div>
      )}

      {/* Hackathon: everything else (show_text, found, reached) is a card at the top; the face stays visible */}
      {display && display.kind !== 'sound' && (
        <div className="absolute top-6 inset-x-0 z-40 flex justify-center px-4 pointer-events-none">
          <div key={`${display.text}-${display.detail}`}
               className="card-in pointer-events-auto max-w-[92%] rounded-2xl px-6 py-4 text-center shadow-2xl cursor-pointer select-none"
               style={{ background: display.color, color: textColorFor(display.color), touchAction: 'manipulation' }}
               onPointerUp={onDismissDisplay}>
            <div className="font-bold leading-tight break-words" style={{ fontSize: `${Math.max(24, 44 * globalScale)}px` }}>
              {display.text}
            </div>
            {display.detail && (
              <div className="mt-1 opacity-80 break-words" style={{ fontSize: `${Math.max(14, 22 * globalScale)}px` }}>
                {display.detail}
              </div>
            )}
          </div>
        </div>
      )}

      {hasCriticalError && (
        <div className="absolute inset-0 flex items-center justify-center z-50 pointer-events-none">
             <div className="text-red-500 font-bold animate-pulse drop-shadow-[0_0_50px_rgba(239,68,68,0.8)]" style={{ fontSize: `${200 * globalScale}px` }}>
                !
             </div>
        </div>
      )}
    </div>
  );
};

export default DogFace;