
import { AppConfig } from './types';

export const DEFAULT_CONFIG: AppConfig = {
  modelName: 'gemini-3.1-flash-live-preview',
  ttsModelName: 'gemini-3.1-flash-tts-preview',
  voiceName: 'Puck',
  // Pupper's personality and rules live on the robot, in lab/prompt.md.
  extraInstructions: '',
  enableVideo: true,
  enableAudio: true,
  // Empty = the address this page was loaded from (the robot server serves the app).
  robotApiUrl: '',
  eyeTrackingSource: 'server',
  cameraSource: 'browser',
  enableAffectiveDialog: true,
  enableProactiveAudio: true,
  captions: true,
  soundAlerts: true,
  objectDetection: true,
};

// Fallback trick list (the monorepo's 13 tricks, now policies), used only if the robot
// can't be reached. At connect time the app asks the robot (GET /animations) which
// tricks are installed.
export const ANIMATION_NAMES = [
  { id: 'twerk', label: 'Twerk', description: 'From standing, drops onto its front legs and wiggles its hips in a rhythmic dance, ending lying down.' },
  { id: 'lie_sit_lie', label: 'Lie-Sit-Lie', description: 'Lies down, sits up on its front legs, then lies back down and stays lying.' },
  { id: 'stand_sit_shake_sit_stand', label: 'Shake', description: 'From standing, sits, shakes a front paw, sits, and stands back up.' },
  { id: 'stand_sit_stand', label: 'Sit', description: 'From standing, sits down and stands back up.' },
  { id: 'upward_dog', label: 'Upward Dog', description: 'Lies down, does the yoga upward dog (chest up, front legs straight), then lies back down.' },
  { id: 'lie_downward_dog', label: 'Downward Dog (lying)', description: 'Lies down, does the yoga downward dog (hips up), then lies back down.' },
  { id: 'stand_downward_dog', label: 'Downward Dog', description: 'From standing, does the yoga downward dog (chest low, hips up) and comes back up to standing.' },
  { id: 'superman', label: 'Superman', description: 'Lies down and stretches its legs out front and back like a flying superhero, then lies still.' },
  { id: 'pee', label: 'Pee', description: 'From standing, lifts a back leg like a dog peeing, then puts it down.' },
  { id: 'push_up', label: 'Push-up', description: 'From standing, lowers and raises its body like push-ups.' },
  { id: 'sneeze', label: 'Sneeze', description: 'From standing, a quick sneezing jerk of the head and body.' },
  { id: 'spider', label: 'Spider', description: 'Lies down and wiggles its legs like a silly spider, ending lying down.' },
  { id: 'swim', label: 'Swim', description: 'Lies down and paddles its legs like swimming, ending lying down.' },
];

/** The robot server's base URL: the configured one, or this page's own origin. */
export function robotBaseUrl(configured: string): string {
  return (configured || window.location.origin).replace(/\/$/, '');
}

export const MODEL_OPTIONS = [
  'gemini-3.1-flash-live-preview',
  'gemini-2.5-flash-native-audio-preview-12-2025',
  'gemini-robotics-er-2-streaming-preview',
];

export const TTS_MODEL_OPTIONS = [
  'gemini-3.1-flash-tts-preview',
  'gemini-2.5-flash-preview-tts',
];

export const VOICE_OPTIONS = ['Puck', 'Charon', 'Kore', 'Fenrir', 'Zephyr'];
