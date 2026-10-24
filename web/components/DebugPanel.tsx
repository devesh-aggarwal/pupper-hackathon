import React, { useRef, useEffect, useState } from 'react';
import { LogEntry, MoveCommand, SystemStatus, CameraSource } from '../types';
import { ANIMATION_NAMES, DEFAULT_CONFIG } from '../constants';
import RobotBridge from '../services/robotBridge';

// --- Icons ported from Rust SVGs ---
const StatusIcon: React.FC<{status: 'active' | 'inactive' | 'loading' | 'unknown'}> = ({ status }) => {
  let fill = '#808080'; // unknown
  if (status === 'active') fill = '#22c55e';
  if (status === 'inactive') fill = '#ef4444';
  if (status === 'loading') fill = '#f59e0b';

  return (
    <svg width="12" height="12" viewBox="0 0 20 20">
      <circle cx="10" cy="10" r="10" fill={fill} />
    </svg>
  );
};

const BatteryIcon: React.FC<{percentage: number | null, charging: boolean}> = ({ percentage, charging }) => {
  const p = percentage || 0;
  let color = '#ef4444';
  if (p > 50) color = '#22c55e';
  else if (p > 20) color = '#fbbf24';
  
  const width = Math.max(2, (p / 100) * 14);

  return (
    <svg width="24" height="12" viewBox="0 0 24 12">
      <rect x="0.5" y="0.5" width="20" height="11" rx="1" stroke={charging ? "#fbbf24" : "#6b7280"} fill="none" strokeWidth="1" />
      <rect x="21" y="3" width="2" height="6" fill={charging ? "#fbbf24" : "#6b7280"} />
      <rect x="2" y="2" width={width} height="8" rx="0.5" fill={color} />
    </svg>
  );
};

// ------------------------------------

interface DebugPanelProps {
  logs: LogEntry[];
  videoElement: HTMLVideoElement | null;
  robotBridge: RobotBridge;
  systemStatus: SystemStatus | null;
  onStatusUpdate: (status: SystemStatus | null) => void;
  cameraSource: CameraSource;
}

const DebugPanel: React.FC<DebugPanelProps> = ({ logs, videoElement, robotBridge, systemStatus, onStatusUpdate, cameraSource }) => {
  const logEndRef = useRef<HTMLDivElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const leftColRef = useRef<HTMLDivElement>(null);

  // Hackathon: who Pupper is talking with, and the one-click reset (lab/hackathon_memory.py)
  const [memory, setMemory] = useState<any>(null);
  const [profileInput, setProfileInput] = useState('');
  const refreshMemory = async () => {
    const res: any = await robotBridge.getMemory();
    setMemory(res && res.memory ? res : null);
  };
  useEffect(() => {
    refreshMemory();
    const interval = setInterval(refreshMemory, 10000);
    return () => clearInterval(interval);
  }, [robotBridge]);
  const resetMemory = async (scope: 'profile' | 'all') => {
    const what = scope === 'all' ? 'ALL memory (every person, reminders, routines)' : `the profile "${memory?.memory?.active}"`;
    if (!window.confirm(`Reset ${what}?`)) return;
    await robotBridge.resetMemory(scope);
    refreshMemory();
  };
  const switchProfile = async () => {
    if (!profileInput.trim()) return;
    await robotBridge.switchProfile(profileInput.trim());
    setProfileInput('');
    refreshMemory();
  };

  // Tricks the robot reports as installed (falls back to the built-in list)
  const [tricks, setTricks] = useState(ANIMATION_NAMES);
  useEffect(() => {
    robotBridge.getAnimations().then((res: any) => {
      if (res && !('success' in res)) {
        setTricks(Object.entries(res).map(([id, t]: [string, any]) => ({
          id, label: id.replace(/_/g, ' '), description: t.description,
        })));
      }
    });
  }, [robotBridge]);

  // Layout State (Percentages)
  const [leftWidth, setLeftWidth] = useState(70);
  const [topHeight, setTopHeight] = useState(60);
  const [isResizingX, setIsResizingX] = useState(false);
  const [isResizingY, setIsResizingY] = useState(false);

  // Server Image State
  const [serverImage, setServerImage] = useState<string | null>(null);
  const lastUrlRef = useRef<string | null>(null);

  // Polling Effect
  useEffect(() => {
    const fetchStatus = async () => {
        const status = await robotBridge.getSystemStatus();
        onStatusUpdate(status);
    };

    fetchStatus(); // Initial
    const interval = setInterval(fetchStatus, 5000); // 5s poll
    
    return () => clearInterval(interval);
  }, [robotBridge, onStatusUpdate]);

  // Polling Effect for Server Image (2Hz)
  useEffect(() => {
    if (cameraSource !== 'server') {
        // Cleanup if switching away from server
        if (lastUrlRef.current) {
            URL.revokeObjectURL(lastUrlRef.current);
            lastUrlRef.current = null;
            setServerImage(null);
        }
        return;
    }

    let active = true;
    const fetchImage = async () => {
        if (!active) return;
        try {
            const blob = await robotBridge.getCameraImage();
            if (!active) return;
            
            if (blob) {
                const url = URL.createObjectURL(blob);
                // Revoke previous to avoid leak
                if (lastUrlRef.current) URL.revokeObjectURL(lastUrlRef.current);
                lastUrlRef.current = url;
                setServerImage(url);
            }
        } catch (e) {
            console.error("Image fetch failed", e);
        }
    };

    fetchImage();
    const interval = setInterval(fetchImage, 500); // 2Hz

    return () => {
        active = false;
        clearInterval(interval);
        if (lastUrlRef.current) {
            URL.revokeObjectURL(lastUrlRef.current);
            lastUrlRef.current = null;
        }
    };
  }, [cameraSource, robotBridge]);

  // Resize Handlers
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isResizingX && containerRef.current) {
        const containerRect = containerRef.current.getBoundingClientRect();
        const newWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100;
        setLeftWidth(Math.max(20, Math.min(80, newWidth)));
      }
      if (isResizingY && leftColRef.current) {
        const colRect = leftColRef.current.getBoundingClientRect();
        const newHeight = ((e.clientY - colRect.top) / colRect.height) * 100;
        setTopHeight(Math.max(20, Math.min(80, newHeight)));
      }
    };

    const handleMouseUp = () => {
      setIsResizingX(false);
      setIsResizingY(false);
      document.body.style.cursor = 'default';
      document.body.style.userSelect = 'auto';
    };

    if (isResizingX || isResizingY) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      document.body.style.userSelect = 'none';
      document.body.style.cursor = isResizingX ? 'col-resize' : 'row-resize';
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizingX, isResizingY]);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Video Element Mounting / Unmounting based on Source
  useEffect(() => {
    if (!videoContainerRef.current) return;

    if (cameraSource === 'browser' && videoElement) {
        // Changed to object-contain for fitting to screen without crop
        videoElement.className = "w-full h-full object-contain bg-black"; 
        videoContainerRef.current.appendChild(videoElement);
    } else {
        // If server mode or disabled, ensure video element is not attached
        if (videoElement && videoElement.parentNode === videoContainerRef.current) {
            videoContainerRef.current.removeChild(videoElement);
        }
    }
  }, [videoElement, cameraSource]);

  const handleManualMove = (cmd: MoveCommand) => {
    robotBridge.queueMove(cmd);
  };
  
  const handleViewSource = () => {
      // The robot server's interactive API page (FastAPI /docs).
      window.open(`${robotBridge.baseUrl}/docs`, '_blank');
  };

  return (
    <div ref={containerRef} className="flex h-full bg-gray-900 text-xs md:text-sm font-mono flex-col select-none">
      
      {/* --- STATUS BAR --- */}
      <div className="h-10 bg-black border-b border-gray-800 flex items-center px-4 justify-between shrink-0">
          {/* Left: Battery & CPU */}
          <div className="flex items-center gap-4">
             <div className="flex items-center gap-2">
                <BatteryIcon 
                    percentage={systemStatus?.battery?.percentage ?? null} 
                    charging={systemStatus?.battery?.is_charging ?? false} 
                />
                <span className={`font-bold ${systemStatus?.battery?.percentage != null && systemStatus.battery.percentage < 20 ? 'text-red-500' : 'text-white'}`}>
                    {systemStatus?.battery?.percentage != null ? `${systemStatus.battery.percentage}%` : '--%'}
                </span>
             </div>
             <div className="w-px h-4 bg-gray-700" />
             <div className="flex items-center gap-2 text-gray-300">
                {/* Safe access for optional properties to prevent crashes on null systemStatus */}
                <span>CPU: {systemStatus?.cpu?.usage?.toFixed(0) ?? '--'}%</span>
                <span>{systemStatus?.cpu?.temperature != null ? `${systemStatus.cpu.temperature.toFixed(0)}°C` : ''}</span>
             </div>
          </div>

          {/* Right: Services & Tools */}
          <div className="flex items-center gap-4">
              <div className="flex items-center gap-1.5">
                  <StatusIcon status={systemStatus?.services?.robot || 'unknown'} />
                  <span className="text-gray-300 font-bold">ROS</span>
              </div>
              <div className="flex items-center gap-1.5">
                  <StatusIcon status={systemStatus?.services?.internet === 'online' ? 'active' : 'inactive'} />
                  <span className="text-gray-300 font-bold">NET</span>
              </div>
              
              <div className="w-px h-4 bg-gray-700 ml-2" />
              
              <button 
                onClick={handleViewSource}
                className="text-gray-400 hover:text-cyan-400 text-[10px] uppercase font-bold transition-colors"
                title="Robot server API"
              >
                API
              </button>
          </div>
      </div>

      <div className="flex flex-1 overflow-hidden p-2">
        {/* Left Column (Video + Logs) */}
        <div 
            ref={leftColRef}
            style={{ width: `${leftWidth}%` }} 
            className="flex flex-col h-full min-w-[200px]"
        >
            {/* Video Area (Top) */}
            <div 
            ref={videoContainerRef} 
            style={{ height: `${topHeight}%` }}
            className="relative bg-black border border-gray-700 rounded overflow-hidden shrink-0"
            >
                <div className="absolute top-1 left-1 bg-red-600/80 text-white px-1 text-[10px] rounded z-10 pointer-events-none backdrop-blur-sm">LIVE CAM</div>
                {cameraSource === 'server' && serverImage && (
                    <img src={serverImage} className="w-full h-full object-contain bg-black" alt="Robot Camera" />
                )}
            </div>

            {/* Horizontal Resizer */}
            <div 
            className="h-2 bg-gray-800 hover:bg-cyan-600 cursor-row-resize flex items-center justify-center shrink-0 transition-colors"
            onMouseDown={() => setIsResizingY(true)}
            >
            <div className="w-8 h-1 bg-gray-600 rounded-full" />
            </div>

            {/* Logs Area (Bottom) */}
            <div className="flex-1 bg-black border border-gray-800 rounded p-2 overflow-y-auto font-mono text-[10px] md:text-xs min-h-0">
            {logs.length === 0 && <div className="text-gray-600 italic">No logs yet...</div>}
            {logs.map(log => (
                <div key={log.id} className="mb-1 break-words border-b border-gray-900/50 pb-0.5 last:border-0">
                <span className="text-gray-500 mr-2 select-text">[{new Date(log.timestamp).toLocaleTimeString()}]</span>
                <span className={`
                    font-bold
                    ${log.type === 'error' ? 'text-red-500' : ''}
                    ${log.type === 'function' ? 'text-yellow-400' : ''}
                    ${log.type === 'info' ? 'text-blue-400' : ''}
                    ${log.type === 'user' ? 'text-green-400' : ''}
                    ${log.type === 'event' ? 'text-purple-400' : ''}
                `}>
                    {log.type.toUpperCase()}:
                </span>
                <span className="text-gray-300 ml-1 select-text">{log.message}</span>
                </div>
            ))}
            <div ref={logEndRef} />
            </div>
        </div>

        {/* Vertical Resizer */}
        <div 
            className="w-2 bg-gray-800 hover:bg-cyan-600 cursor-col-resize flex items-center justify-center shrink-0 transition-colors"
            onMouseDown={() => setIsResizingX(true)}
        >
            <div className="h-8 w-1 bg-gray-600 rounded-full" />
        </div>

        {/* Right Column: Controls */}
        <div className="flex-1 min-w-[200px] bg-gray-800 p-2 rounded border border-gray-700 flex flex-col gap-2 h-full">
            
            {/* Header & Main Actions */}
            <div className="flex gap-2 shrink-0">
                <button 
                    onClick={() => robotBridge.queueActivateWalking()} 
                    className="flex-1 bg-green-900/50 border border-green-800 hover:bg-green-800 p-2 rounded text-green-100 transition-colors text-center"
                    title="Activate Motors"
                >
                    ACTIVATE
                </button>
                <button 
                    onClick={() => handleManualMove({vx: 0, vy: 0, wz: 0, duration: 0})} 
                    className="flex-1 bg-red-900/50 border border-red-800 hover:bg-red-800 p-2 rounded text-red-100 transition-colors font-bold text-center"
                >
                    STOP
                </button>
            </div>

            <div className="h-px bg-gray-700 shrink-0" />

            {/* Hackathon: Memory */}
            <div className="shrink-0">
                <div className="text-gray-400 text-[10px] uppercase font-bold mb-1">Memory</div>
                {memory ? (
                  <div className="text-[10px] text-gray-300 space-y-1">
                    <div>
                      talking with <span className="text-cyan-300 font-bold">{memory.memory.active}</span>
                      {' '}({(memory.memory.profiles?.[memory.memory.active]?.facts?.length ?? 0)} facts,
                      {' '}{Object.keys(memory.memory.profiles?.[memory.memory.active]?.objects ?? {}).length} things)
                      {' '}· {memory.memory.reminders?.length ?? 0} reminders · {Object.keys(memory.memory.routines ?? {}).length} routines
                    </div>
                    <div className="text-gray-500 truncate" title={Object.keys(memory.memory.profiles ?? {}).join(', ')}>
                      people: {Object.keys(memory.memory.profiles ?? {}).join(', ') || 'none yet'}
                    </div>
                    <div className="flex gap-1">
                      <input
                        value={profileInput}
                        onChange={e => setProfileInput(e.target.value)}
                        onKeyDown={e => { if (e.key === 'Enter') switchProfile(); }}
                        placeholder="switch to name..."
                        className="flex-1 bg-gray-900 border border-gray-700 rounded px-1.5 py-1 text-[10px] text-white outline-none focus:border-cyan-500"
                      />
                      <button onClick={switchProfile} className="bg-gray-700 hover:bg-cyan-900 hover:text-cyan-200 px-2 rounded text-[10px] transition-colors">Switch</button>
                    </div>
                    <div className="flex gap-1">
                      <button onClick={() => resetMemory('profile')} className="flex-1 bg-gray-700 hover:bg-yellow-900 hover:text-yellow-200 p-1.5 rounded text-[10px] transition-colors">Reset this person</button>
                      <button onClick={() => resetMemory('all')} className="flex-1 bg-gray-700 hover:bg-red-900 hover:text-red-200 p-1.5 rounded text-[10px] transition-colors">Reset all</button>
                    </div>
                  </div>
                ) : (
                  <div className="text-[10px] text-gray-600 italic">memory unavailable (robot server not running?)</div>
                )}
            </div>

            <div className="h-px bg-gray-700 shrink-0" />

            {/* Follow Mode */}
            <div className="shrink-0">
                <div className="text-gray-400 text-[10px] uppercase font-bold mb-1">Follow Mode</div>
                <div className="flex gap-1">
                    <button 
                        onClick={() => robotBridge.setFollowMode(true)}
                        className="flex-1 bg-gray-700 hover:bg-cyan-900 hover:text-cyan-200 p-1.5 rounded text-[10px] transition-colors"
                    >
                        ON
                    </button>
                    <button 
                        onClick={() => robotBridge.setFollowMode(false)}
                        className="flex-1 bg-gray-700 hover:bg-gray-600 p-1.5 rounded text-[10px] transition-colors"
                    >
                        OFF
                    </button>
                </div>
            </div>

            <div className="h-px bg-gray-700 shrink-0" />

            {/* Arrow Keys Grid */}
            <div className="shrink-0 flex flex-col items-center gap-1 py-1">
                {/* Up */}
                <button 
                    onClick={() => handleManualMove({vx: 0.5, vy: 0, wz: 0, duration: 1})} 
                    className="w-12 h-10 bg-blue-900 border border-blue-800 hover:bg-blue-800 rounded flex items-center justify-center text-lg active:scale-95 transition-all shadow-lg"
                >
                    ▲
                </button>
                
                {/* Left Down Right */}
                <div className="flex gap-1">
                    <button 
                        onClick={() => handleManualMove({vx: 0, vy: 0, wz: 45, duration: 1})} 
                        className="w-12 h-10 bg-blue-900 border border-blue-800 hover:bg-blue-800 rounded flex items-center justify-center text-lg active:scale-95 transition-all shadow-lg"
                    >
                        ◄
                    </button>
                    <button 
                        onClick={() => handleManualMove({vx: -0.5, vy: 0, wz: 0, duration: 1})} 
                        className="w-12 h-10 bg-blue-900 border border-blue-800 hover:bg-blue-800 rounded flex items-center justify-center text-lg active:scale-95 transition-all shadow-lg"
                    >
                        ▼
                    </button>
                    <button 
                        onClick={() => handleManualMove({vx: 0, vy: 0, wz: -45, duration: 1})} 
                        className="w-12 h-10 bg-blue-900 border border-blue-800 hover:bg-blue-800 rounded flex items-center justify-center text-lg active:scale-95 transition-all shadow-lg"
                    >
                        ►
                    </button>
                </div>
            </div>

            <div className="h-px bg-gray-700 shrink-0" />
            
            <div className="text-gray-400 font-bold text-[10px] uppercase tracking-wider shrink-0">Animations</div>
            
            {/* Scrollable Animation List */}
            <div className="flex flex-col gap-1 overflow-y-auto flex-1 min-h-0 pr-1">
                {tricks.map(anim => (
                <button 
                    key={anim.id}
                    onClick={() => robotBridge.queueAnimation(anim.id)}
                    className="bg-gray-700/50 p-2 rounded text-[10px] hover:bg-gray-700 text-left transition-all border border-gray-600/50 hover:border-gray-500 group"
                >
                    <div className="font-bold text-cyan-400 group-hover:text-cyan-300">{anim.label}</div>
                    <div className="text-gray-500 group-hover:text-gray-400 text-[9px] truncate">{anim.description}</div>
                </button>
                ))}
            </div>
        </div>
      </div>
    </div>
  );
};

export default DebugPanel;