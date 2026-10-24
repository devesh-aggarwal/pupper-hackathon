/**
 * Demo mode (open the app with ?demo): the real face, awake, with no robot and no Gemini.
 * It listens on the real microphone: a sound in the room perks the ears up and shows
 * in the corner, as on the robot. Nothing is scripted.
 */
export const DEMO = typeof window !== "undefined" && new URLSearchParams(window.location.search).has("demo");

export interface DemoHooks {
  setConnected: (on: boolean) => void;
  setHearing: (text: string | null) => void;
}

export function runDemo(h: DemoHooks): () => void {
  let stopped = false;
  let micStop = () => {};
  h.setConnected(true);

  // Loudness over a slowly adapting room level, checked every 100 ms.
  navigator.mediaDevices?.getUserMedia({ audio: true }).then(stream => {
    if (stopped) { stream.getTracks().forEach(t => t.stop()); return; }
    const ctx = new AudioContext();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    let background = -55, lastLoud = 0;
    const timer = window.setInterval(() => {
      analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (const v of buf) sum += v * v;
      const db = 20 * Math.log10(Math.sqrt(sum / buf.length) + 1e-9);
      const now = performance.now();
      if (db - background >= 10 && db >= -50) lastLoud = now;
      else background += (db - background) * (db < background ? 0.2 : 0.02);
      h.setHearing(now - lastLoud < 600 ? "Sound" : null);
    }, 100);
    micStop = () => { clearInterval(timer); stream.getTracks().forEach(t => t.stop()); ctx.close(); };
  }).catch(() => { /* no mic: the face just stays awake */ });

  return () => { stopped = true; micStop(); };
}
