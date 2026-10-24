/**
 * Hearing for people who can't: measures how loud the microphone is and reports
 * sudden loud moments (a knock, a doorbell, something falling) to the robot, which
 * decides whether to alert (lab/hackathon_alerts.py: alerts off, too quiet for the
 * sensitivity, or Pupper's own motors, and nothing happens). The browser only measures.
 *
 * It is fed the same 16 kHz PCM blocks that go to Gemini (GeminiLiveService.setupAudioInput).
 * Loudness is in dB relative to a slowly adapting background level, so a quiet room
 * and a busy hackathon hall both work. Pupper's own voice through the speaker is
 * ignored (isSpeaking) so it never alerts on itself.
 */
export class SoundMonitor {
  private background = -55; // dBFS, adapts slowly
  private loudSince = 0;
  private peakAbove = 0;
  private peakLevel = -100;
  private lastReport = 0;
  private lastSpeaking = 0;

  constructor(
    private onLoud: (aboveDb: number, levelDb: number, durationS: number) => void,
    private isSpeaking: () => boolean,
    private minAboveDb = 10,
    private minLevelDb = -50,
  ) {}

  feed(samples: Float32Array, now: number = performance.now()) {
    if (this.isSpeaking()) this.lastSpeaking = now;
    const muted = now - this.lastSpeaking < 400; // Pupper's own voice, and its echo
    // Loudest 64 ms block in this buffer (a short bang would vanish in a 256 ms average).
    const block = 1024;
    let maxDb = -100;
    for (let start = 0; start < samples.length; start += block) {
      const end = Math.min(samples.length, start + block);
      let sum = 0;
      for (let i = start; i < end; i++) sum += samples[i] * samples[i];
      const db = 20 * Math.log10(Math.sqrt(sum / (end - start)) + 1e-9);
      if (db > maxDb) maxDb = db;
    }
    const above = maxDb - this.background;
    const loud = !muted && above >= this.minAboveDb && maxDb >= this.minLevelDb;

    if (loud) {
      if (!this.loudSince) this.loudSince = now;
      this.peakAbove = Math.max(this.peakAbove, above);
      this.peakLevel = Math.max(this.peakLevel, maxDb);
      if (now - this.loudSince > 1500 && now - this.lastReport > 4000) {
        this.report(now); // a long sound (an alarm): report while it goes on
        this.loudSince = now;
      }
      return;
    }
    if (this.loudSince && !muted) this.report(now);
    this.loudSince = 0;
    this.peakAbove = 0;
    this.peakLevel = -100;
    if (!muted) {
      // Quiet frames set the background: down quickly, up slowly, so one bang doesn't raise it.
      const rate = maxDb < this.background ? 0.2 : 0.02;
      this.background += (maxDb - this.background) * rate;
    }
  }

  private report(now: number) {
    if (now - this.lastReport < 1500) return;
    this.lastReport = now;
    const duration = Math.max(0.06, (now - this.loudSince) / 1000);
    this.onLoud(Math.round(this.peakAbove), Math.round(this.peakLevel), Math.round(duration * 10) / 10);
  }
}
