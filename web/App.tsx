import React, { useState, useEffect, useRef, useCallback } from 'react';
import { ConnectionState, AppConfig, LogEntry, SystemStatus, ScreenDisplay } from './types';
import { DEFAULT_CONFIG } from './constants';
import DogFace from './components/DogFace';
import DebugPanel from './components/DebugPanel';
import Settings from './components/Settings';
import { GeminiLiveService } from './services/geminiLive';
import RobotBridge from './services/robotBridge';
import { RobotEvents } from './services/robotEvents';
import { DEMO, runDemo } from './services/demo';

// Icons
const CogIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.09a2 2 0 0 1-1-1.74v-.47a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>
);
const TerminalIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/></svg>
);
const FaceIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
);
const PowerIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18.36 6.64a9 9 0 1 1-12.73 0"/><line x1="12" y1="2" x2="12" y2="12"/></svg>
);
const MoonIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
);

const App: React.FC = () => {
  const [view, setView] = useState<'face' | 'debug'>('face');
  const [showSettings, setShowSettings] = useState(false);
  const [connState, setConnState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  
  // Load API Key from localStorage
  const [apiKey, setApiKey] = useState(() => {
    return localStorage.getItem('pupster_api_key') || '';
  });

  // Load Config from localStorage
  const [config, setConfig] = useState<AppConfig>(() => {
    const saved = localStorage.getItem('pupster_config');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        // The system prompt comes from the robot (lab/prompt.md), never from saved settings.
        delete parsed.systemInstruction;
        return { ...DEFAULT_CONFIG, ...parsed };
      } catch (e) {
        console.error("Failed to parse saved config", e);
      }
    }
    return DEFAULT_CONFIG;
  });
  
  // Persist API Key changes
  useEffect(() => {
    localStorage.setItem('pupster_api_key', apiKey);
  }, [apiKey]);

  // Persist Config changes
  useEffect(() => {
    localStorage.setItem('pupster_config', JSON.stringify(config));
  }, [config]);

  // Runtime State
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [audioLevel, setAudioLevel] = useState(0);
  // Hackathon: big words on the screen (show_text, reminders, sound alerts) and captions
  const [display, setDisplay] = useState<ScreenDisplay | null>(null);
  const [caption, setCaption] = useState<string | null>(null);
  const displayTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const showDisplay = useCallback((d: ScreenDisplay | null) => {
    if (displayTimerRef.current) clearTimeout(displayTimerRef.current);
    setDisplay(d);
    if (d) displayTimerRef.current = setTimeout(() => setDisplay(null), Math.max(1, d.seconds) * 1000);
  }, []);
  
  // Create a persistent video element
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const hiddenVideoContainerRef = useRef<HTMLDivElement>(null);

  // UI Visibility State (for auto-hiding footer)
  const [isUiVisible, setIsUiVisible] = useState(true);
  const idleTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  
  useEffect(() => {
    const v = document.createElement('video');
    v.autoplay = true;
    v.muted = true;
    v.playsInline = true;
    setVideoElement(v);
    return () => {
        v.srcObject = null;
    };
  }, []);

  // When in Face mode, park the video element in the hidden container so it stays active
  useEffect(() => {
    if (view === 'face' && videoElement && hiddenVideoContainerRef.current) {
        hiddenVideoContainerRef.current.appendChild(videoElement);
    }
  }, [view, videoElement]);

  // Activity Monitor for Auto-Hiding UI
  useEffect(() => {
    const resetIdleTimer = () => {
      setIsUiVisible(true);
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
      
      // Only set auto-hide timer if we are in face mode
      if (view === 'face') {
        idleTimerRef.current = setTimeout(() => {
          setIsUiVisible(false);
        }, 5000); // 5 seconds
      }
    };

    // Attach listeners
    window.addEventListener('mousemove', resetIdleTimer);
    window.addEventListener('mousedown', resetIdleTimer);
    window.addEventListener('touchstart', resetIdleTimer);
    window.addEventListener('keydown', resetIdleTimer);

    // Initial trigger
    resetIdleTimer();

    return () => {
      window.removeEventListener('mousemove', resetIdleTimer);
      window.removeEventListener('mousedown', resetIdleTimer);
      window.removeEventListener('touchstart', resetIdleTimer);
      window.removeEventListener('keydown', resetIdleTimer);
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current);
    };
  }, [view]);

  const geminiRef = useRef<GeminiLiveService | null>(null);
  // Persistent bridge instance for polling
  const bridgeRef = useRef<RobotBridge>(new RobotBridge(config.robotApiUrl));

  useEffect(() => {
    bridgeRef.current = new RobotBridge(config.robotApiUrl);
  }, [config.robotApiUrl]);

  const addLog = useCallback((entry: LogEntry) => {
    setLogs(prev => [...prev.slice(-99), entry]); // Keep last 100 logs
  }, []);

  // Hackathon: the robot's events paint the screen even while Pupper sleeps (a reminder
  // at 9 still shows). The Gemini session has its own poller that tells Gemini.
  useEffect(() => {
    const bridge = bridgeRef.current;
    bridge.getDisplay().then(d => { if (d) showDisplay(d); });
    const events = new RobotEvents(bridge, e => { if (e.display) showDisplay(e.display); });
    events.start();
    return () => events.stop();
  }, [config.robotApiUrl, showDisplay]);

  // Hackathon: what Pupper hears right now (only while sound alerts are on), for a corner of the face.
  const [hearing, setHearing] = useState<string | null>(null);
  useEffect(() => {
    if (DEMO) return;  // the demo script sets what Pupper hears
    const interval = setInterval(async () => setHearing(await bridgeRef.current.getHearingNow()), 500);
    return () => clearInterval(interval);
  }, [config.robotApiUrl]);

  // Reminders: every 5 s, ask the robot for due ones (set with the set_reminder tool,
  // lab/reminders.py). Connected: Gemini is told, so Pupper says it in its own voice.
  // Not connected: wait a minute for the session to come back (the kiosk reconnects by
  // itself), then say it with the browser's speech. Either way it shows on screen, and
  // the robot is told it was delivered; unconfirmed, it is due again a minute later.
  // Reminders stay on screen until someone taps (anywhere), says "got it" (Gemini calls
  // dismiss_reminder; the robot lists it as dismissed), or 2 minutes pass.
  const [reminderBanners, setReminderBanners] = useState<{ id: number; message: string }[]>([]);
  const bannerTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reminderTickRef = useRef<(() => void) | null>(null);
  const dismissReminders = useCallback(() => {
    if (bannerTimerRef.current) clearTimeout(bannerTimerRef.current);
    setReminderBanners(prev => {
      if (prev.length) bridgeRef.current.dismissReminders(prev.map(b => b.id));  // so dismiss_reminder knows
      return [];
    });
    // Clear any show_text card too (Gemini often echoes the reminder there), never a sound alert
    setDisplay(d => (d && d.kind !== 'sound' ? null : d));
  }, []);
  useEffect(() => {
    let busy = false;
    const tick = async () => {
      if (busy) return;
      busy = true;
      try {
        const gemini = geminiRef.current;
        const connected = !!gemini?.isConnected;
        const res = await bridgeRef.current.claimReminders(connected ? 0 : 60);
        const dismissed = res?.dismissed ?? [];
        if (dismissed.length) {
          setReminderBanners(prev => {
            const left = prev.filter(b => !dismissed.includes(b.id));
            if (left.length < prev.length && !left.length) setDisplay(d => (d && d.kind !== 'sound' ? null : d));
            return left;
          });
        }
        for (const r of res?.due ?? []) {
          let how = 'screen only';
          if (gemini && await gemini.announce(r.announce)) {
            how = 'gemini';
          } else if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
            window.speechSynthesis.speak(new SpeechSynthesisUtterance(r.speech));
            how = 'browser speech';
          }
          setReminderBanners(prev => [...prev.filter(b => b.id !== r.id), { id: r.id, message: r.message }]);
          if (bannerTimerRef.current) clearTimeout(bannerTimerRef.current);
          bannerTimerRef.current = setTimeout(() => setReminderBanners([]), 120000);
          addLog({ id: `rem-${r.id}-${Date.now()}`, timestamp: Date.now(), type: 'info',
                   message: `Reminder #${r.id} due: "${r.message}" (told via ${how})` });
          await bridgeRef.current.reminderDelivered(r.id, how);
        }
      } finally {
        busy = false;
      }
    };
    reminderTickRef.current = tick;
    const interval = setInterval(tick, 5000);
    return () => { clearInterval(interval); reminderTickRef.current = null; };
  }, [addLog]);

  // Demo mode (?demo): a scripted scene, no robot and no Gemini (services/demo.ts)
  useEffect(() => {
    if (!DEMO) return;
    return runDemo({
      setConnected: on => setConnState(on ? ConnectionState.CONNECTED : ConnectionState.DISCONNECTED),
      setAudioLevel, setCaption, setHearing, showDisplay, setReminders: setReminderBanners,
    });
  }, [showDisplay]);

  // Background System Polling Loop (Slow, for critical battery alerts)
  useEffect(() => {
    const interval = setInterval(async () => {
        const status = await bridgeRef.current.getSystemStatus();
        setSystemStatus(status);
        // Do not log here to avoid spam. DebugPanel will handle its own fast polling.
    }, 60000); // Poll every 60s
    return () => clearInterval(interval);
  }, [addLog]);

  const handleConnect = async () => {
    if (!apiKey) {
      addLog({ id: 'err-key', timestamp: Date.now(), type: 'error', message: 'API Key missing' });
      setShowSettings(true);
      return;
    }
    
    if (!videoElement) {
        addLog({ id: 'err-vid', timestamp: Date.now(), type: 'error', message: 'Video initialization failed' });
        return;
    }

    try {
      setConnState(ConnectionState.CONNECTING);
      
      const service = new GeminiLiveService(
        config,
        addLog,
        (level) => setAudioLevel(level),
        () => setConnState(ConnectionState.DISCONNECTED)
      );

      service.onCaption = setCaption;
      await service.connect(apiKey, config, videoElement);
      // "Got it" -> dismiss_reminder: clear the screen now, not at the next 5 s check.
      service.onToolResult = (name) => {
        if (name === 'dismiss_reminder') reminderTickRef.current?.();
      };
      geminiRef.current = service;
      setConnState(ConnectionState.CONNECTED);
    } catch (err: any) {
      console.error(err);
      addLog({ id: 'conn-err', timestamp: Date.now(), type: 'error', message: err.message });
      setConnState(ConnectionState.ERROR);
    }
  };

  const handleDisconnect = async () => {
    if (geminiRef.current) {
      try {
        await geminiRef.current.robotBridge.queueDeactivate();
      } catch (e) { console.error(e); }
      await geminiRef.current.disconnect();
      geminiRef.current = null;
    }
    
    if (videoElement) {
        videoElement.srcObject = null;
    }
    
    setConnState(ConnectionState.DISCONNECTED);
  };

  return (
    <div className="h-screen w-screen bg-black flex flex-col font-sans text-gray-100 overflow-hidden relative">
      
      <div 
        ref={hiddenVideoContainerRef} 
        style={{ position: 'absolute', top: 0, left: 0, width: '1px', height: '1px', opacity: 0, overflow: 'hidden', pointerEvents: 'none' }} 
      />

      <main 
        className="flex-1 relative overflow-hidden"
      >
        {view === 'face' ? (
          <DogFace 
            isListening={connState === ConnectionState.CONNECTED}
            isSpeaking={audioLevel > 0.05} 
            audioLevel={audioLevel}
            systemStatus={systemStatus}
            config={config}
            videoElement={videoElement}
            display={display}
            caption={caption}
            hearing={hearing}
            onDismissDisplay={() => showDisplay(null)}
          />
        ) : (
          <DebugPanel 
            logs={logs} 
            videoElement={videoElement}
            robotBridge={geminiRef.current ? (geminiRef.current as any).robotBridge : bridgeRef.current} 
            systemStatus={systemStatus}
            onStatusUpdate={setSystemStatus}
            cameraSource={config.cameraSource}
          />
        )}

        {reminderBanners.length > 0 && (
          // A card at the top, not a takeover: the face stays visible. Tap the card to
          // dismiss (pointerup fires for touch, pen and mouse; manipulation stops zoom).
          <div className="absolute top-6 inset-x-0 z-50 flex justify-center px-4 pointer-events-none">
            <div
              id="reminder-card"
              role="button"
              aria-label="Dismiss reminder"
              onPointerUp={dismissReminders}
              onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ' || e.key === 'Escape') dismissReminders(); }}
              tabIndex={0}
              style={{ touchAction: 'manipulation' }}
              className="card-in pointer-events-auto max-w-[92%] bg-white text-gray-900 rounded-2xl px-6 py-4 shadow-2xl cursor-pointer select-none flex items-start gap-4"
            >
              <div className="text-3xl leading-none mt-1">⏰</div>
              <div className="text-left">
                <div className="text-xs font-semibold uppercase tracking-wider text-amber-600">Reminder</div>
                {reminderBanners.map(b => (
                  <div key={b.id} className="text-2xl font-semibold leading-snug">{b.message}</div>
                ))}
                <div className="text-sm mt-1 text-gray-500">Tap to dismiss</div>
              </div>
            </div>
          </div>
        )}

        {connState === ConnectionState.DISCONNECTED && !DEMO && (
           <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm z-10 pointer-events-auto">
              <button 
                onClick={handleConnect}
                className="bg-cyan-500 hover:bg-cyan-400 text-black font-bold py-4 px-8 rounded-full shadow-[0_0_30px_rgba(34,211,238,0.5)] transform hover:scale-105 transition-all text-xl flex items-center gap-2"
              >
                <PowerIcon /> WAKE PUPPER
              </button>
           </div>
        )}
      </main>

      <footer className={`
        h-16 flex items-center justify-between px-4 z-20 transition-all duration-500 ease-in-out
        ${view === 'face' ? 'absolute bottom-0 left-0 right-0 border-t-0 bg-gray-900/60 backdrop-blur-sm' : 'relative bg-gray-900 border-t border-gray-800'}
        ${view === 'face' && !isUiVisible ? 'translate-y-full opacity-0 pointer-events-none' : 'translate-y-0 opacity-100 pointer-events-auto'}
      `}>
        <div className="flex items-center space-x-3 w-32">
           <div className={`w-3 h-3 rounded-full ${
             connState === ConnectionState.CONNECTED ? 'bg-green-500 shadow-[0_0_10px_#22c55e]' : 
             connState === ConnectionState.CONNECTING ? 'bg-yellow-500 animate-pulse' : 'bg-red-500'
           }`} />
           <span className="text-xs font-mono text-gray-400 uppercase">{connState}</span>
        </div>

        <div className="flex bg-gray-800 rounded-lg p-1">
          <button 
            onClick={() => setView('face')}
            className={`p-2 rounded ${view === 'face' ? 'bg-gray-700 text-cyan-400' : 'text-gray-500'}`}
          >
            <FaceIcon />
          </button>
          <button 
            onClick={() => setView('debug')}
            className={`p-2 rounded ${view === 'debug' ? 'bg-gray-700 text-cyan-400' : 'text-gray-500'}`}
          >
            <TerminalIcon />
          </button>
        </div>

        <div className="flex items-center space-x-2 w-32 justify-end">
          {connState === ConnectionState.CONNECTED ? (
             <button 
               onClick={handleDisconnect}
               title="Sleep / Disconnect"
               className="bg-red-900/50 hover:bg-red-900 text-red-200 p-2 rounded-full border border-red-800 transition-colors"
             >
               <PowerIcon />
             </button>
          ) : (
            <div className="text-gray-500 flex items-center gap-1 mr-2" title="Sleeping">
              <MoonIcon />
              <span className="text-xs font-medium">SLEEPING</span>
            </div>
          )}
          <button 
            onClick={() => {
              setShowSettings(true);
              if (document.fullscreenElement) {
                document.exitFullscreen().catch(e => console.error(e));
              }
            }}
            className="p-2 text-gray-400 hover:text-white"
            title="Settings"
          >
            <CogIcon />
          </button>
        </div>
      </footer>

      {showSettings && (
        <Settings 
          apiKey={apiKey} 
          setApiKey={setApiKey} 
          config={config} 
          setConfig={setConfig} 
          onClose={() => setShowSettings(false)} 
        />
      )}
    </div>
  );
};

export default App;