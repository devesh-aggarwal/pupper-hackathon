

export enum ConnectionState {
  DISCONNECTED = 'DISCONNECTED',
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  ERROR = 'ERROR',
}

export enum RobotMode {
  IDLE = 'IDLE',
  WALKING = 'WALKING',
  ANIMATING = 'ANIMATING',
}

export interface LogEntry {
  id: string;
  timestamp: number;
  type: 'info' | 'error' | 'function' | 'user' | 'model' | 'event';
  message: string;
}

export type EyeTrackingSource = 'mouse' | 'server' | 'mediapipe';
export type CameraSource = 'browser' | 'server';

export interface AppConfig {
  modelName: string;
  ttsModelName?: string;
  voiceName: string;
  extraInstructions: string; // added after the robot's prompt (lab/prompt.md)
  enableVideo: boolean;
  enableAudio: boolean;
  robotApiUrl: string; // URL for the FastAPI bridge
  eyeTrackingSource: EyeTrackingSource;
  cameraSource: CameraSource;
  enableAffectiveDialog: boolean;
  enableProactiveAudio: boolean;
  // Hackathon features (see README, "Hackathon features")
  captions: boolean; // show what Pupper says as text on the face
  soundAlerts: boolean; // measure the mic and report loud sounds to the robot
  objectDetection: boolean; // run the on-device object detector on camera frames
}

/** Big words on Pupper's screen (show_text, a reminder, a sound alert). */
export interface ScreenDisplay {
  text: string;
  detail: string;
  color: string;
  seconds: number;
  kind?: string; // "sound" takes over the whole screen; everything else is a card
}

/** Something that happened on the robot (GET /events): a reminder, a sound, a search result. */
export interface RobotEvent {
  id: number;
  time: number;
  clock: string;
  kind: string;
  text: string;
  for_gemini: string;
  display: ScreenDisplay | null;
  data: Record<string, unknown>;
}

export interface MoveCommand {
  vx: number; // forward/back
  vy: number; // left/right
  wz: number; // turn
  duration: number;
}

/** A reminder that is due (POST /reminders/claim on the robot, robot/reminder_store.py). */
export interface DueReminder {
  id: number;
  message: string;
  due_at: string;
  late_s: number;
  repeat: string;
  announce: string; // what Gemini is told, so Pupper says it
  speech: string; // what the browser says itself if Gemini isn't connected
}

export interface SystemStatus {
  battery: {
    percentage: number | null;
    is_charging: boolean;
    voltage: number | null;
  };
  cpu: {
    usage: number;
    temperature: number | null;
  };
  services: {
    robot: 'active' | 'inactive' | 'unknown';
    internet: 'online' | 'offline' | 'unknown';
  };
}