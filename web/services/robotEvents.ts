import RobotBridge from "./robotBridge";
import { RobotEvent } from "../types";

/**
 * Events are things that happened on the robot that nobody asked for right then: a
 * reminder came due, the microphone heard a bang, a search found the cup, a walk
 * stopped early. The robot server collects them (GET /events?after=<id>); this polls
 * once a second and hands each new one to onEvent. Two of these run at once: the app's
 * (to paint the screen, even while Pupper sleeps) and the Gemini session's (to tell
 * Gemini). Each keeps its own cursor, and a fresh one starts at "now" (after=-1), so
 * old events are never replayed.
 */
export class RobotEvents {
  private after = -1;
  private timer: number | null = null;
  private busy = false;

  constructor(private bridge: RobotBridge, private onEvent: (event: RobotEvent) => void) {}

  start(intervalMs = 1000) {
    this.stop();
    this.poll();
    this.timer = window.setInterval(() => this.poll(), intervalMs);
  }

  stop() {
    if (this.timer !== null) clearInterval(this.timer);
    this.timer = null;
  }

  private async poll() {
    if (this.busy) return;
    this.busy = true;
    try {
      const res = await this.bridge.getEvents(this.after);
      if (res && typeof res.last === "number") {
        if (this.after >= 0) for (const e of res.events) this.onEvent(e);
        this.after = res.last;
      }
    } finally {
      this.busy = false;
    }
  }
}
