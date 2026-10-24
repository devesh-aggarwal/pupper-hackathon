import RobotBridge from './robotBridge';
import { AppConfig, LogEntry } from '../types';
import { SoundMonitor } from './soundMonitor';
import { OnDeviceObjectDetector } from './objectDetector';

/**
 * Pupper's brain is robot/gemma_brain.py (Gemma, no cloud), never Gemini Live, on this branch. It
 * listens and talks on the robot itself, so the face only shows it: listening, a mouth that moves
 * while it talks, and the caption, from what it posts (GET /brain/state). Everything else the
 * Gemini session does for the robot runs here too: the sound monitor on the mic and the on-device
 * object detector on the robot's camera. Reminders the app claims go to the brain to say
 * (POST /brain/announce).
 */
export const LOCAL_BRAIN = true;  // the app never connects Gemini (no WAKE PUPPER button, no key)

export class LocalBrainService {
  /** The brain was heard from in the last 5 s (it posts every second). */
  isConnected = false;
  /** Text on the screen (captions, cards): only for a deaf or hard-of-hearing person (the robot knows). */
  onText: (on: boolean) => void = () => {};
  private timers: number[] = [];
  private captionTimer: number | null = null;
  private speaking = false;
  private sound: SoundMonitor | null = null;
  private detector: OnDeviceObjectDetector | null = null;
  private inputCtx: AudioContext | null = null;
  private stream: MediaStream | null = null;

  constructor(
    private bridge: RobotBridge,
    private config: AppConfig,
    private onLog: (entry: LogEntry) => void,
    private onAudioLevel: (level: number) => void,
    private onCaption: (text: string | null) => void,
    private onConnected: (on: boolean) => void,
  ) {}

  async start() {
    this.timers.push(window.setInterval(() => this.poll(), 200));
    // The voice plays on the robot, so the mouth can't follow its loudness: it moves in syllables.
    const t0 = performance.now();
    this.timers.push(window.setInterval(() => {
      if (!this.speaking) return this.onAudioLevel(0);
      const t = (performance.now() - t0) / 1000;
      this.onAudioLevel(0.08 + Math.abs(Math.sin(t * 11)) * (0.6 + 0.4 * Math.sin(t * 2.3)) * 0.45);
    }, 50));
    if (this.config.soundAlerts) await this.startSound();
    if (this.config.enableVideo && this.config.objectDetection) this.startDetector();
  }

  stop() {
    this.timers.forEach(t => clearInterval(t));
    this.timers = [];
    if (this.captionTimer) clearTimeout(this.captionTimer);
    this.stream?.getTracks().forEach(t => t.stop());
    this.inputCtx?.close().catch(() => {});
    this.detector?.close();
  }

  /** A due reminder: the brain says it in its own voice. */
  async announce(text: string): Promise<boolean> {
    if (!this.isConnected) return false;
    const res = await this.bridge.announceToBrain(text);
    if (res?.success) this.log('info', `Told the brain: ${text}`);
    return !!res?.success;
  }

  private async poll() {
    const s = await this.bridge.getBrainState();
    const alive = !!s && s.age_s !== null && s.age_s < 5;
    if (alive !== this.isConnected) {
      this.isConnected = alive;
      this.onConnected(alive);
      this.log(alive ? 'info' : 'error', alive ? 'Local brain (Gemma) is up' : 'Local brain (Gemma) is not answering');
    }
    this.speaking = alive && !!s.speaking;
    this.onText(!!s?.text);
    if (alive && s.caption && this.config.captions) {
      if (this.captionTimer) { clearTimeout(this.captionTimer); this.captionTimer = null; }
      this.onCaption(s.caption);
    } else if (!this.captionTimer) {
      // Like Gemini's captions: the last one stays up a few seconds after Pupper stops talking.
      this.captionTimer = window.setTimeout(() => this.onCaption(null), 5000);
    }
  }

  private async startSound() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      this.inputCtx = new AudioContext({ sampleRate: 16000 });
      const source = this.inputCtx.createMediaStreamSource(this.stream);
      const proc = this.inputCtx.createScriptProcessor(4096, 1, 1);
      this.sound = new SoundMonitor((above, level, duration) => {
        this.log('event', `Loud sound: +${above} dB for ${duration} s (reported to the robot)`);
        this.bridge.postSound(above, level, duration);
      }, () => this.speaking);
      proc.onaudioprocess = e => this.sound?.feed(e.inputBuffer.getChannelData(0));
      source.connect(proc);
      proc.connect(this.inputCtx.destination);
      this.log('info', 'Sound monitor on the microphone');
    } catch (e: any) {
      this.log('error', `No microphone for the sound monitor: ${e?.message || e}`);
    }
  }

  private startDetector() {
    this.detector = new OnDeviceObjectDetector();
    this.detector.load()
      .then(() => this.log('info', 'On-device object detector ready (MediaPipe EfficientDet-Lite0)'))
      .catch(e => this.log('error', `Object detector failed to load: ${e?.message || e}`));
    this.timers.push(window.setInterval(async () => {
      if (!this.detector?.ready) return;
      try {
        const blob = await this.bridge.getCameraImage();
        const frame = blob ? await this.detector.detectBlob(blob) : null;
        if (frame) await this.bridge.postDetections(frame);
      } catch (e) {
        // a dropped frame is fine
      }
    }, 1000));
  }

  private log(type: LogEntry['type'], message: string) {
    this.onLog({ id: `local-${Date.now()}-${Math.random()}`, timestamp: Date.now(), type, message });
  }
}
