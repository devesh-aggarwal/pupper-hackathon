import { ObjectDetector, FilesetResolver } from "@mediapipe/tasks-vision";

/**
 * Pupper's on-device eyes for obstacles and lost things: MediaPipe's EfficientDet-Lite0
 * (80 everyday classes: person, chair, cup, bottle, cell phone, laptop, sports ball ...)
 * running in the browser. GeminiLiveService runs it on the camera frames it already
 * fetches for Gemini and posts the result to the robot (POST /vision/detections), where
 * lab/hackathon_vision.py refuses walks into things and runs find's looks.
 *
 * Boxes are fractions of the frame: [x, y, w, h], x to the right, y down.
 */
export interface Detection {
  label: string;
  score: number;
  box: [number, number, number, number];
}

export interface DetectionFrame {
  detections: Detection[];
  width: number;
  height: number;
}

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite";

export class OnDeviceObjectDetector {
  private detector: ObjectDetector | null = null;
  private loading: Promise<void> | null = null;
  private busy = false;

  get ready() { return this.detector !== null; }

  /** Loads the model (about 4 MB, cached by the browser). GPU first, then CPU. */
  load(): Promise<void> {
    if (!this.loading) {
      this.loading = (async () => {
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        for (const delegate of ["GPU", "CPU"] as const) {
          try {
            this.detector = await ObjectDetector.createFromOptions(vision, {
              baseOptions: { modelAssetPath: MODEL_URL, delegate },
              runningMode: "IMAGE",
              scoreThreshold: 0.4,
              maxResults: 8,
            });
            return;
          } catch (e) {
            console.warn(`[ObjectDetector] ${delegate} failed`, e);
          }
        }
        throw new Error("object detector could not start on GPU or CPU");
      })();
    }
    return this.loading;
  }

  /** Detect in a JPEG from the robot's camera (the frames Gemini gets). */
  async detectBlob(blob: Blob): Promise<DetectionFrame | null> {
    if (!this.detector || this.busy) return null;
    this.busy = true;
    try {
      const bitmap = await createImageBitmap(blob);
      try {
        return this.convert(this.detector.detect(bitmap), bitmap.width, bitmap.height);
      } finally {
        bitmap.close();
      }
    } catch (e) {
      return null;
    } finally {
      this.busy = false;
    }
  }

  /** Detect in the browser's own camera video. */
  detectVideo(video: HTMLVideoElement): DetectionFrame | null {
    if (!this.detector || this.busy || !video.videoWidth) return null;
    this.busy = true;
    try {
      return this.convert(this.detector.detect(video), video.videoWidth, video.videoHeight);
    } catch (e) {
      return null;
    } finally {
      this.busy = false;
    }
  }

  private convert(result: any, width: number, height: number): DetectionFrame {
    const detections: Detection[] = [];
    for (const d of result?.detections ?? []) {
      const cat = d.categories?.[0];
      const bb = d.boundingBox;
      if (!cat || !bb || !width || !height) continue;
      detections.push({
        label: String(cat.categoryName || cat.displayName || "thing").toLowerCase(),
        score: Math.round(cat.score * 100) / 100,
        box: [bb.originX / width, bb.originY / height, bb.width / width, bb.height / height]
          .map(v => Math.round(Math.max(0, Math.min(1, v)) * 1000) / 1000) as [number, number, number, number],
      });
    }
    return { detections, width, height };
  }

  close() {
    this.detector?.close();
    this.detector = null;
    this.loading = null;
  }
}
