
import { GoogleGenAI, LiveServerMessage, Modality, Type, FunctionDeclaration } from '@google/genai';
import { createPcmBlob, decodeAudioData, blobToBase64 } from './audioUtils';
import { AppConfig, LogEntry, RobotEvent, ScreenDisplay } from '../types';
import RobotBridge from './robotBridge';
import { SoundMonitor } from './soundMonitor';
import { OnDeviceObjectDetector } from './objectDetector';
import { RobotEvents } from './robotEvents';

/**
 * Gemini's tools and system prompt come from the robot (GET /tools): they are
 * written in Python, in lab/ (lab/prompt.md, and the @tool functions). Every tool call is
 * sent back to the robot (POST /tools/<name>), and its {success, message} goes
 * to Gemini as the result. Only `speak` lives here: text-only models use it to
 * talk through the browser's speech synthesis.
 */
const SPEAK_TOOL: FunctionDeclaration = {
  name: 'speak',
  description: 'Speak a message out loud to the user via voice synthesis. Use this tool whenever you want to talk, reply, narrate actions, or converse.',
  parameters: {
    type: Type.OBJECT,
    properties: { text: { type: Type.STRING, description: 'The text message to synthesize and speak aloud.' } },
    required: ['text'],
  },
};

export function isAudioOutputModel(modelName: string): boolean {
  const name = modelName.toLowerCase();
  if (name.includes('robotics') || name.includes('er-2') || name.includes('er2')) {
    return false;
  }
  return true;
}

export class GeminiLiveService {
  private ai: GoogleGenAI | null = null;
  private sessionPromise: Promise<any> | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private inputScriptProcessor: ScriptProcessorNode | null = null;
  private mediaStream: MediaStream | null = null;
  private nextStartTime = 0;
  private videoInterval: number | null = null;
  private volumeInterval: number | null = null;
  private queuedSources: AudioBufferSourceNode[] = [];
  private connected = false;
  private config: AppConfig;
  // Hackathon: the robot's events (reminders, sounds, searches) go to Gemini as turns;
  // the sound monitor and the on-device object detector feed the robot.
  private events: RobotEvents | null = null;
  private sound: SoundMonitor | null = null;
  private detector: OnDeviceObjectDetector | null = null;
  private lastOutputLevel = 0;
  private caption = '';
  private captionTimer: number | null = null;
  private heard = '';  // the person's words this turn (Gemini's input transcription), for the session log
  private said = '';   // Pupper's words this turn

  public robotBridge: RobotBridge;

  // Callbacks
  onLog: (entry: LogEntry) => void;
  onAudioLevel: (level: number) => void;
  onDisconnect: () => void;
  /** Captions: what Pupper is saying, as text (null when it is done). */
  onCaption: (text: string | null) => void = () => {};
  /** Called after every robot tool call (e.g. App refreshes the reminders after dismiss_reminder). */
  onToolResult?: (name: string, result: any) => void;

  constructor(
    config: AppConfig,
    onLog: (entry: LogEntry) => void,
    onAudioLevel: (level: number) => void,
    onDisconnect: () => void
  ) {
    this.config = config;
    this.robotBridge = new RobotBridge(config.robotApiUrl);
    this.onLog = onLog;
    this.onAudioLevel = onAudioLevel;
    this.onDisconnect = onDisconnect;
    this.queuedSources = [];
  }

  async connect(apiKey: string, config: AppConfig, videoElement: HTMLVideoElement) {
    this.config = config;
    this.ai = new GoogleGenAI({ apiKey: apiKey });

    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    
    // Create Analyser for output audio to visualize speaker sound
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.1;
    this.analyser.connect(this.audioContext.destination);

    this.startOutputVolumeMonitoring();

    const inputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      const tracks: MediaStreamTrack[] = [];

      // Try acquiring audio
      if (config.enableAudio) {
        try {
          const audioStream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
            }
          });
          tracks.push(...audioStream.getAudioTracks());
          this.log('info', 'Microphone connected successfully');
        } catch (audioErr: any) {
          this.log('error', `Microphone access failed: ${audioErr.message || audioErr}`);
          console.warn('Microphone error:', audioErr);
        }
      }

      // Try acquiring video
      if (config.enableVideo && config.cameraSource === 'browser') {
        try {
          const videoStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 } }
          });
          tracks.push(...videoStream.getVideoTracks());
          this.log('info', 'Camera connected successfully');
        } catch (videoErr: any) {
          this.log('error', `Camera access failed: ${videoErr.message || videoErr}`);
          console.warn('Camera error:', videoErr);
        }
      }

      if (tracks.length > 0) {
        this.mediaStream = new MediaStream(tracks);
      }
    } else {
      this.log('error', 'MediaDevices API not available. Ensure you are on HTTPS or http://localhost');
      console.warn('MediaDevices API not available. Ensure you are on HTTPS or http://localhost');
    }

    if (config.enableVideo && config.cameraSource === 'browser' && videoElement && this.mediaStream && this.mediaStream.getVideoTracks().length > 0) {
      videoElement.srcObject = this.mediaStream;
      videoElement.play().catch(e => console.error("Video play failed", e));
    }

    const modelName = config.modelName === 'gemini-2.5-flash-native-audio-preview'
      ? 'gemini-2.5-flash-native-audio-preview-12-2025'
      : config.modelName;
    const isAudioOutput = isAudioOutputModel(modelName);

    // Ask the robot for Gemini's system prompt and tools (lab/prompt.md, the tools in lab/).
    const spec: any = await this.robotBridge.getTools();
    const robotTools: FunctionDeclaration[] = Array.isArray(spec?.tools) ? spec.tools : [];
    if (robotTools.length) {
      this.log('info', `Robot offers ${robotTools.length} tools: ${robotTools.map(t => t.name).join(', ')}`);
    } else {
      this.log('error', `Could not get tools from the robot (${spec?.message ?? 'no tools'}); Gemini can talk but not move.`);
    }
    let systemInstruction = [spec?.prompt, config.extraInstructions].filter(Boolean).join('\n\n');
    if (!isAudioOutput) {
      systemInstruction += '\n\nIMPORTANT: You do not have direct voice audio output. To speak out loud to the user or narrate your actions, you MUST invoke the "speak" function tool with your response text.';
    }

    const functionDeclarations = isAudioOutput ? robotTools : [...robotTools, SPEAK_TOOL];

    const liveConfig: any = {
      responseModalities: isAudioOutput ? [Modality.AUDIO] : [Modality.TEXT],
      systemInstruction: systemInstruction,
      tools: [{ functionDeclarations }],
      contextWindowCompression: { slidingWindow: {} },
    };
    if (isAudioOutput && config.captions) {
      // Hackathon: the transcript of Pupper's own speech, shown on the face for people who can't hear it.
      liveConfig.outputAudioTranscription = {};
    }
    // Hackathon: both sides of the conversation go to the robot's session log (POST /transcript).
    liveConfig.inputAudioTranscription = {};
    if (config.soundAlerts) {
      this.sound = new SoundMonitor((above, level, duration) => {
        this.log('event', `Loud sound: +${above} dB for ${duration} s (reported to the robot)`);
        this.robotBridge.postSound(above, level, duration);
      }, () => this.lastOutputLevel > 0.05);
    }
    if (config.enableVideo && config.objectDetection) {
      this.detector = new OnDeviceObjectDetector();
      this.detector.load()
        .then(() => this.log('info', 'On-device object detector ready (MediaPipe EfficientDet-Lite0)'))
        .catch(e => this.log('error', `Object detector failed to load: ${e?.message || e}`));
    }

    if (isAudioOutput) {
      liveConfig.speechConfig = {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: config.voiceName } },
      };
      // Gemini 3.1 Flash Live rejects these Gemini 2.5-only options during
      // session setup, which otherwise looks like a successful connection
      // followed immediately by an unexplained WebSocket close.
      const supportsExperimentalAudioFeatures = !modelName.toLowerCase().includes('gemini-3.1');
      if (supportsExperimentalAudioFeatures && config.enableAffectiveDialog) {
        liveConfig.enableAffectiveDialog = true;
      }
      if (supportsExperimentalAudioFeatures && config.enableProactiveAudio) {
        liveConfig.proactivity = { proactiveAudio: true };
      }
    }

    // Connect and store the promise
    this.log('info', `Connecting to ${modelName} (Audio Output: ${isAudioOutput ? 'Native' : 'Flash TTS via speak tool'})...`);
    this.sessionPromise = this.ai.live.connect({
      model: modelName,
      config: liveConfig,
      callbacks: {
        onopen: () => {
          this.connected = true;
          this.log('info', 'Session connected');
          if (this.mediaStream) {
            this.setupAudioInput(inputCtx, this.mediaStream);
          }
          if (config.enableVideo) {
            this.setupVideoInput(videoElement, config.cameraSource);
          }
          // Hackathon: things that happen on the robot reach Gemini as turns.
          this.events = new RobotEvents(this.robotBridge, e => this.handleRobotEvent(e));
          this.events.start();
        },
        onmessage: async (msg: LiveServerMessage) => {
          if (!this.connected) return;
          await this.handleMessage(msg);
        },
        onclose: (event: CloseEvent) => {
          const details = [
            event.code ? `code ${event.code}` : '',
            event.reason || '',
          ].filter(Boolean).join(': ');
          this.log('error', `Session closed${details ? ` (${details})` : ''}`);
          this.cleanup();
          this.onDisconnect();
        },
        onerror: (err: ErrorEvent) => {
          const details = err.message || err.error?.message || String(err.error || 'WebSocket error');
          this.log('error', `Session error: ${details}`);
          console.log(err);
        }
      }
    });

    // Wait for connection to ensure validity
    await this.sessionPromise;
  }

  private startOutputVolumeMonitoring() {
    if (this.volumeInterval) clearInterval(this.volumeInterval);
    this.volumeInterval = window.setInterval(() => {
      if (!this.analyser) return;
      
      const array = new Uint8Array(this.analyser.fftSize);
      this.analyser.getByteTimeDomainData(array);
      
      let sum = 0;
      for (let i = 0; i < array.length; i++) {
        const a = (array[i] - 128) / 128;
        sum += a * a;
      }
      const rms = Math.sqrt(sum / array.length);
      // Boost factor to make mouth movement more visible
      this.lastOutputLevel = Math.min(1.0, rms * 3);
      this.onAudioLevel(this.lastOutputLevel);
    }, 50);
  }

  private setupAudioInput(ctx: AudioContext, stream: MediaStream) {
    const source = ctx.createMediaStreamSource(stream);
    this.inputScriptProcessor = ctx.createScriptProcessor(4096, 1, 1);
    
    this.inputScriptProcessor.onaudioprocess = (e) => {
      if (!this.connected || !this.sessionPromise) return;

      const inputData = e.inputBuffer.getChannelData(0);
      this.sound?.feed(inputData);
      const pcmBlob = createPcmBlob(inputData);
      
      this.sessionPromise.then(session => {
        if (!this.connected) return;
        try {
           session.sendRealtimeInput({ 
             audio: { 
               data: pcmBlob.data, 
               mimeType: pcmBlob.mimeType 
             } 
           });
        } catch(err) {
           // Ignore errors if session is closing
        }
      });
    };

    source.connect(this.inputScriptProcessor);
    this.inputScriptProcessor.connect(ctx.destination);
  }

  private setupVideoInput(videoEl: HTMLVideoElement, source: 'browser' | 'server') {
    if (this.videoInterval) clearInterval(this.videoInterval);
    
    if (source === 'browser') {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        this.videoInterval = window.setInterval(async () => {
          if (!this.connected || !this.sessionPromise || !ctx || !videoEl.videoWidth) return;
          
          canvas.width = videoEl.videoWidth * 0.5;
          canvas.height = videoEl.videoHeight * 0.5;
          ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
          this.detectAndReport(() => this.detector?.detectVideo(videoEl) ?? null);
          
          canvas.toBlob(async (blob) => {
            if (!blob || !this.connected) return;
            const base64 = await blobToBase64(blob);
            this.sessionPromise?.then(session => {
                if (!this.connected) return;
                try {
                    session.sendRealtimeInput({ 
                      video: { 
                        mimeType: 'image/jpeg', 
                        data: base64 
                      }
                    })
                } catch(err) {}
            });
          }, 'image/jpeg', 0.6);
        }, 1000);
    } else {
        // Server source
        this.videoInterval = window.setInterval(async () => {
             if (!this.connected || !this.sessionPromise) return;
             const blob = await this.robotBridge.getCameraImage();
             if (blob && this.connected) {
                 this.detectAndReport(() => this.detector?.detectBlob(blob) ?? null);
                 const base64 = await blobToBase64(blob);
                 this.sessionPromise?.then(session => {
                    if (!this.connected) return;
                    try {
                        session.sendRealtimeInput({ 
                          video: { 
                            mimeType: 'image/jpeg', 
                            data: base64 
                          }
                        })
                    } catch(err) {}
                });
             }
        }, 1000); // 2Hz
    }
  }

  private async playAudioBase64(audioData: string): Promise<void> {
    if (!this.audioContext || !this.analyser) return;
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
    this.nextStartTime = Math.max(this.nextStartTime, this.audioContext.currentTime);
    const buffer = await decodeAudioData(audioData, this.audioContext);
    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.analyser);
    source.start(this.nextStartTime);
    this.queuedSources.push(source);
    this.nextStartTime += buffer.duration;

    // Streaming callers must be able to enqueue the next chunk immediately;
    // waiting for playback to finish here would recreate full-response latency.
    source.onended = () => {
      const idx = this.queuedSources.indexOf(source);
      if (idx !== -1) {
        this.queuedSources.splice(idx, 1);
      }
    };
  }

  private async synthesizeAndPlaySpeech(text: string): Promise<void> {
    try {
      if (!this.ai) return;
      const configuredModel = this.config.ttsModelName || "gemini-3.1-flash-tts-preview";
      const ttsModel = configuredModel === "gemini-2.5-flash-preview"
        ? "gemini-2.5-flash-preview-tts"
        : configuredModel;
      const supportsStreaming = ttsModel.startsWith("gemini-3.1-");
      this.log("info", "Synthesizing speech via " + ttsModel + (supportsStreaming ? " (streaming)..." : "..."));

      const request: any = {
        model: ttsModel,
        contents: [{ parts: [{ text }] }],
        config: {
          responseModalities: ["AUDIO"],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: {
                voiceName: this.config.voiceName || "Puck",
              }
            }
          }
        }
      };

      let receivedAudio = false;
      if (supportsStreaming) {
        const responseStream = await this.ai.models.generateContentStream(request);
        for await (const chunk of responseStream) {
          const parts = chunk.candidates?.[0]?.content?.parts || [];
          for (const part of parts) {
            const audioData = part.inlineData?.data;
            if (audioData) {
              receivedAudio = true;
              await this.playAudioBase64(audioData);
            }
          }
        }
      } else {
        const response = await this.ai.models.generateContent(request);
        const parts = response.candidates?.[0]?.content?.parts || [];
        for (const part of parts) {
          const audioData = part.inlineData?.data;
          if (audioData) {
            receivedAudio = true;
            await this.playAudioBase64(audioData);
          }
        }
      }

      if (!receivedAudio) {
        console.warn("No audio data returned by TTS model, using browser speech fallback");
        this.speakWithWebSpeech(text);
      }
    } catch (err: any) {
      this.log("error", "TTS error: " + (err.message || err) + ". Using browser speech fallback.");
      this.speakWithWebSpeech(text);
    }
  }

  private speakWithWebSpeech(text: string): void {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      let interval: number | null = null;
      utterance.onstart = () => {
        interval = window.setInterval(() => {
          this.onAudioLevel(0.2 + Math.random() * 0.3);
        }, 100);
      };
      const stopVisual = () => {
        if (interval) clearInterval(interval);
        this.onAudioLevel(0);
      };
      utterance.onend = stopVisual;
      utterance.onerror = stopVisual;
      window.speechSynthesis.speak(utterance);
    }
  }

  /** Run the on-device detector on a frame (if it is ready) and tell the robot what it sees. */
  private async detectAndReport(detect: () => Promise<any> | any) {
    if (!this.detector?.ready || !this.connected) return;
    try {
      const frame = await detect();
      if (frame && this.connected) await this.robotBridge.postDetections(frame);
    } catch (e) {
      // a dropped frame is fine
    }
  }

  /** An event from the robot: show it in the log and give it to Gemini as a turn. */
  private handleRobotEvent(event: RobotEvent) {
    this.log('event', `${event.kind}: ${event.text}`);
    if (!event.for_gemini || !this.connected || !this.sessionPromise) return;
    this.sessionPromise.then(session => {
      if (!this.connected) return;
      try {
        session.sendClientContent({ turns: [{ role: 'user', parts: [{ text: event.for_gemini }] }], turnComplete: true });
      } catch (err) {
        // Session may have closed
      }
    });
  }

  private showCaption(text: string, done: boolean) {
    if (this.captionTimer) { clearTimeout(this.captionTimer); this.captionTimer = null; }
    this.caption = text;
    this.onCaption(text || null);
    if (done || !text) {
      this.captionTimer = window.setTimeout(() => { this.caption = ''; this.onCaption(null); }, 5000);
    }
  }

  private async handleMessage(message: LiveServerMessage) {
    // Check for native audio output
    const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
    if (audioData && this.audioContext && this.analyser) {
      await this.playAudioBase64(audioData);
    }

    // Check for text output (useful for ER2 streaming / text models)
    const textPart = message.serverContent?.modelTurn?.parts?.find((p: any) => p.text)?.text;
    if (textPart) {
      this.log('model', `Text: ${textPart}`);
    }

    // Hackathon captions: the transcript of what Pupper says, for people who can't hear it.
    const transcript = (message.serverContent as any)?.outputTranscription?.text;
    if (transcript && this.config.captions) {
      this.showCaption(this.caption + transcript, false);
    } else if (textPart && this.config.captions && !isAudioOutputModel(this.config.modelName)) {
      this.showCaption(this.caption + textPart, false);
    }
    if (transcript) this.said += transcript;
    const heard = (message.serverContent as any)?.inputTranscription?.text;
    if (heard) this.heard += heard;
    if (message.serverContent?.turnComplete) {
      if (this.heard.trim()) this.robotBridge.postTranscript('person', this.heard.trim());
      if (this.said.trim()) this.robotBridge.postTranscript('pupper', this.said.trim());
      this.heard = ''; this.said = '';
    }
    if (message.serverContent?.turnComplete && this.caption) {
      this.showCaption(this.caption, true);
    }

    if (message.toolCall) {
      for (const fc of message.toolCall.functionCalls) {
        this.log('function', `Call: ${fc.name}(${JSON.stringify(fc.args)})`);
        let result: any = { success: true };
        
        try {
          // Helper args
          const args = fc.args as any;

          switch (fc.name) {
            case 'speak':
            case 'say':
            case 'narrate':
              if (isAudioOutputModel(this.config.modelName)) {
                result = { success: false, error: 'Native-audio models must speak through their Live API audio response' };
                break;
              }
              const speechText = args.text || args.message || args.content || '';
              if (speechText) {
                this.log('model', `Pupper: "${speechText}"`);
                await this.synthesizeAndPlaySpeech(speechText);
                result = { success: true, spoken: true, text: speechText };
              } else {
                result = { success: false, error: 'No text provided to speak' };
              }
              break;
            default:
              // Every other tool is defined and run on the robot (the tools in lab/).
              result = await this.robotBridge.callTool(fc.name, args ?? {});
          }
        } catch (e: any) {
          result = { error: e.message };
        }

        this.log(result?.success === false ? 'error' : 'function', `Result: ${fc.name} -> ${result?.message ?? JSON.stringify(result)}`);
        try { this.onToolResult?.(fc.name, result); } catch (e) { console.error(e); }
        if (this.connected && this.sessionPromise) {
          this.sessionPromise.then(session => {
            if (!this.connected) return;
            try {
              session.sendToolResponse({
                functionResponses: {
                  id: fc.id,
                  name: fc.name,
                  response: { result }
                }
              });
            } catch (err) {
              // Session may have closed between check and send
            }
          });
        }
      }
    }

    if (message.serverContent?.interrupted) {
      this.log('info', 'Model interrupted');
      this.showCaption('', true);
      this.queuedSources.forEach(s => {
        try { s.stop(); } catch (e) {}
      });
      this.queuedSources = [];
      this.nextStartTime = 0;
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    }
  }

  get isConnected(): boolean {
    return this.connected && this.sessionPromise !== null;
  }

  /**
   * Tell Gemini something that doesn't come from the person (a reminder that is due),
   * so it answers out loud right away. True if it reached the session.
   */
  async announce(text: string): Promise<boolean> {
    if (!this.isConnected) return false;
    try {
      const session = await this.sessionPromise;
      if (!this.connected) return false;
      try {
        // Supported for the whole session on every Live model; turnComplete makes Gemini answer now.
        session.sendClientContent({ turns: [{ role: 'user', parts: [{ text }] }], turnComplete: true });
      } catch (err) {
        session.sendRealtimeInput({ text });
      }
      this.log('info', `Told Gemini: ${text}`);
      return true;
    } catch (err: any) {
      this.log('error', `Could not tell Gemini about a reminder: ${err?.message || err}`);
      return false;
    }
  }

  /**
   * Stop all intervals and null out session/resources.
   * Called both on explicit disconnect and on server-side onclose.
   */
  private cleanup() {
    this.connected = false;
    this.events?.stop();
    this.events = null;
    this.detector?.close();
    this.detector = null;
    this.sound = null;
    if (this.captionTimer) { clearTimeout(this.captionTimer); this.captionTimer = null; }
    this.caption = '';
    this.onCaption(null);

    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }

    if (this.inputScriptProcessor) {
      this.inputScriptProcessor.disconnect();
      this.inputScriptProcessor = null;
    }
    if (this.videoInterval) {
      clearInterval(this.videoInterval);
      this.videoInterval = null;
    }
    if (this.volumeInterval) {
      clearInterval(this.volumeInterval);
      this.volumeInterval = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    this.analyser = null;
    this.queuedSources = [];
    this.nextStartTime = 0;
  }

  public async disconnect() {
    // Set flag first to stop all in-flight sends immediately
    this.connected = false;
    this.cleanup();

    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }

    // Critical: Close the session to allow reconnection
    if (this.sessionPromise) {
      try {
        const session = await this.sessionPromise;
        await session.close();
      } catch (e) {
        console.error("Error closing session", e);
      }
    }

    this.sessionPromise = null;
    this.ai = null;
  }

  private log(type: LogEntry['type'], message: string) {
    this.onLog({
      id: Math.random().toString(36),
      timestamp: Date.now(),
      type,
      message
    });
  }
}
