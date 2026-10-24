import { DueReminder, MoveCommand, RobotEvent, ScreenDisplay, SystemStatus } from "../types";
import { DetectionFrame } from "./objectDetector";
import { robotBaseUrl } from "../constants";

// This service talks to the Python FastAPI backend running on the robot
class RobotBridge {
  readonly baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = robotBaseUrl(baseUrl);
  }

  private async request(endpoint: string, method: 'GET' | 'POST', body?: any, returnBlob: boolean = false) {
    try {
      const options: RequestInit = {
        method,
        headers: body && method === 'POST' ? { 'Content-Type': 'application/json' } : {},
      };

      if (body && method === 'POST') {
        options.body = JSON.stringify(body);
      }

      const response = await fetch(`${this.baseUrl}${endpoint}`, options);

      if (!response.ok) {
        throw new Error(`robot server returned HTTP ${response.status} for ${endpoint}`);
      }
      
      if (returnBlob) {
        return await response.blob();
      }
      return await response.json();
    } catch (e) {
      // Suppress log spam for polling
      if(endpoint !== '/system/status' && endpoint !== '/camera/image') {
         console.error(`[RobotBridge] Error calling ${endpoint}`, e);
      }
      return { success: false, message: `robot unreachable: ${(e as Error)?.message || e}` };
    }
  }

  async getSystemStatus(): Promise<SystemStatus | null> {
    const res = await this.request('/system/status', 'GET');
    if (res && !res.error) {
      console.log("Returned value from /system/status is", JSON.stringify(res, null, 2));
      return res as SystemStatus;
    }
    return null;
  }
  
  async getCameraImage(): Promise<Blob | null> {
      try {
          const blob = await this.request('/camera/image', 'GET', undefined, true);
          if (blob instanceof Blob) {
              return blob;
          }
      } catch (e) {}
      return null;
  }

  async queueActivateWalking() {
    return this.request('/walking/activate', 'POST');
  }

  async queueDeactivate() {
    return this.request('/walking/deactivate', 'POST');
  }

  async queueMove(cmd: MoveCommand) {
    // Map internal MoveCommand to API MoveVelocityRequest
    const body = {
      forward_backward_velocity: cmd.vx,
      right_left_velocity: cmd.vy,
      turning_velocity: cmd.wz,
      duration: cmd.duration
    };
    return this.request('/move/velocity', 'POST', body);
  }

  async moveDirection(heading: number, speed: number, duration: number) {
    return this.request('/move/direction', 'POST', { heading, speed, duration });
  }

  async queueStop() {
    return this.request('/queue/stop', 'POST');
  }

  async queueWait(duration: number) {
    return this.request('/queue/wait', 'POST', { duration });
  }

  async queueReset() {
    return this.request('/queue/reset', 'POST');
  }

  async immediateStop() {
    return this.request('/stop/immediate', 'POST');
  }

  async queueAnimation(name: string) {
    return this.request('/animation', 'POST', { animation_name: name });
  }

  /** Gemini's system prompt and tool declarations (lab/prompt.md, the tools in lab/). */
  async getTools() {
    return this.request('/tools', 'GET');
  }

  /** Run one of Gemini's tools on the robot. Returns {success, message}. */
  async callTool(name: string, args: Record<string, unknown>) {
    return this.request(`/tools/${encodeURIComponent(name)}`, 'POST', args);
  }

  /** Due reminders for this page to tell the person; each is hidden from other claimers for a minute. */
  async claimReminders(minLateS: number = 0): Promise<{ due?: DueReminder[]; dismissed?: number[] }> {
    return this.request(`/reminders/claim?min_late_s=${minLateS}`, 'POST');
  }

  /** A tap dismissed these from the screen; tell the robot (so dismiss_reminder knows). */
  async dismissReminders(ids: number[]) {
    return this.request('/reminders/dismiss', 'POST', { ids });
  }

  /** The person was told (how: 'gemini', 'browser speech', ...): done, or moved to its next time. */
  async reminderDelivered(id: number, how: string) {
    return this.request(`/reminders/${id}/delivered`, 'POST', { how });
  }

  async getLimits() {
    return this.request('/limits', 'GET');
  }

  async getQueueStatus() {
    return this.request('/queue/status', 'GET');
  }

  async interactiveAction(action: string) {
    return this.request('/interactive_action', 'POST', { action });
  }

  async getInteractiveActions() {
    return this.request('/interactive_actions', 'GET');
  }

  async getAnimations() {
    return this.request('/animations', 'GET');
  }

  async setVolume(volume: number) {
    return this.request('/system/volume', 'POST', { volume });
  }

  async checkMode() {
    return this.request('/system/mode', 'GET');
  }

  async setFollowMode(enabled: boolean) {
    const endpoint = enabled ? '/following/activate' : '/following/deactivate';
    return this.request(endpoint, 'POST');
  }

  async analyzeCamera(prompt: string) {
    return this.request('/camera/analyze', 'POST', { prompt });
  }

  // ---- Hackathon features: events, the screen, hearing, vision, memory

  /** Events since `after` (an id); after=-1 only returns the latest id. Null if unreachable. */
  async getEvents(after: number): Promise<{ events: RobotEvent[]; last: number } | null> {
    const res = await this.request(`/events?after=${after}`, 'GET');
    return res && Array.isArray(res.events) ? res : null;
  }

  /** What is on Pupper's screen right now, or null. */
  async getDisplay(): Promise<ScreenDisplay | null> {
    const res = await this.request('/display', 'GET');
    return res && res.display ? (res.display as ScreenDisplay) : null;
  }

  /** What Pupper hears right now (YAMNet's top label), or null for silence or when sound alerts are off. */
  async getHearingNow(): Promise<string | null> {
    const res = await this.request('/hearing/now', 'GET');
    return res && typeof res.label === 'string' ? res.label : null;
  }

  /** What was said, for the robot's session log (who: 'person' or 'pupper'). */
  async postTranscript(who: 'person' | 'pupper', text: string) {
    return this.request('/transcript', 'POST', { who, text });
  }

  /** The sound monitor heard something loud; the robot decides whether to alert. */
  async postSound(aboveDb: number, levelDb: number, durationS: number) {
    return this.request('/alerts/sound', 'POST', { above_db: aboveDb, level_db: levelDb, duration_s: durationS });
  }

  /** The local brain (robot/gemma_brain.py): listening, speaking, the caption, and how fresh that is. */
  async getBrainState(): Promise<{ listening: boolean; speaking: boolean; caption: string | null; age_s: number | null; local: boolean } | null> {
    return this.request('/brain/state', 'GET');
  }

  /** A due reminder for the local brain to say (it arrives there as an event). */
  async announceToBrain(text: string) {
    return this.request('/brain/announce', 'POST', { text });
  }

  /** What the on-device object detector sees in the current camera frame. */
  async postDetections(frame: DetectionFrame) {
    return this.request('/vision/detections', 'POST', frame);
  }

  async getMemory() {
    return this.request('/memory', 'GET');
  }

  async resetMemory(scope: 'profile' | 'all') {
    return this.request('/memory/reset', 'POST', { scope });
  }

  async switchProfile(name: string) {
    return this.request('/memory/profile', 'POST', { name });
  }
}

export default RobotBridge;