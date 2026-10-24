import { ScreenDisplay } from "../types";

/**
 * Demo mode (open the app with ?demo): a scripted scene on the real face and screen UI,
 * with no robot and no Gemini. It loops: wake up, listen, talk with captions, a
 * reminder card, a "found it" card, a sound alert, then fall asleep.
 */
export const DEMO = typeof window !== "undefined" && new URLSearchParams(window.location.search).has("demo");

export interface DemoHooks {
  setConnected: (on: boolean) => void;
  setAudioLevel: (level: number) => void;
  setCaption: (text: string | null) => void;
  setHearing: (text: string | null) => void;
  showDisplay: (d: ScreenDisplay | null) => void;
  setReminders: (r: { id: number; message: string }[]) => void;
}

export function runDemo(h: DemoHooks): () => void {
  let stopped = false;
  let talkTimer: number | null = null;
  const wait = (ms: number) => new Promise<void>(r => setTimeout(r, ms));

  // Fake a voice: a wobbly audio level for as long as the caption takes to read.
  const say = async (text: string) => {
    h.setHearing(null);
    h.setCaption(text);
    const ms = 900 + text.length * 55;
    const t0 = performance.now();
    talkTimer = window.setInterval(() => {
      const t = (performance.now() - t0) / 1000;
      const syllable = Math.abs(Math.sin(t * 11)) * (0.6 + 0.4 * Math.sin(t * 2.3));
      h.setAudioLevel(0.08 + syllable * 0.45);
    }, 50);
    await wait(ms);
    if (talkTimer !== null) clearInterval(talkTimer);
    talkTimer = null;
    h.setAudioLevel(0);
    await wait(500);
    h.setCaption(null);
  };
  const listen = async (ms: number) => { h.setHearing("Speech"); await wait(ms); h.setHearing(null); };

  const scene = async () => {
    while (!stopped) {
      h.setConnected(false);              // sleepy
      await wait(3500); if (stopped) return;
      h.setConnected(true);               // wakes up, neutral
      await wait(1500);
      await listen(2200);                 // someone talks: ears perk up
      await say("Hi! I'm Pupper. How can I help?");
      await listen(2500);
      await say("Sure. I'll look for your cup.");
      await wait(1800);
      h.showDisplay({ text: "CUP", detail: "found: on the table, left", color: "#22c55e", seconds: 5, kind: "found" });
      await say("Found it! Your cup is on the table, to my left.");
      await wait(2500);
      h.setReminders([{ id: 1, message: "Take your medicine" }]);
      await say("Reminder: it's time to take your medicine.");
      await wait(3000);
      h.setReminders([]);
      await wait(1500);
      h.showDisplay({ text: "DOORBELL", detail: "heard just now", color: "#ef4444", seconds: 4, kind: "sound" });
      await wait(4500);
      await listen(2000);
      await say("Okay, I'll be here if you need me.");
      await wait(2500);
    }
  };
  scene();

  return () => {
    stopped = true;
    if (talkTimer !== null) clearInterval(talkTimer);
  };
}
