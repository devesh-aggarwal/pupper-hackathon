
import React from 'react';
import { AppConfig, EyeTrackingSource, CameraSource } from '../types';
import { MODEL_OPTIONS, TTS_MODEL_OPTIONS, VOICE_OPTIONS } from '../constants';
import { isAudioOutputModel } from '../services/geminiLive';

interface SettingsProps {
  apiKey: string;
  setApiKey: (key: string) => void;
  config: AppConfig;
  setConfig: React.Dispatch<React.SetStateAction<AppConfig>>;
  onClose: () => void;
}

const Settings: React.FC<SettingsProps> = ({ apiKey, setApiKey, config, setConfig, onClose }) => {
  const isRoboticsER = !isAudioOutputModel(config.modelName);

  return (
    <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4">
      <div className="bg-gray-800 rounded-lg max-w-md w-full p-6 shadow-2xl border border-gray-700">
        <h2 className="text-xl font-bold mb-4 text-white">System Configuration</h2>

        <div className="space-y-4 max-h-[70vh] overflow-y-auto pr-2">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Gemini API Key</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your API Key"
              className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-white focus:border-cyan-500 outline-none"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Model</label>
            <select
              value={config.modelName}
              onChange={(e) => setConfig({...config, modelName: e.target.value})}
              className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-white"
            >
              {MODEL_OPTIONS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>
            {isRoboticsER && (
              <p className="mt-1 text-[11px] text-cyan-400">
                Robotics ER 2 Streaming uses text modality; audio is synthesized via Flash TTS and the <code className="bg-gray-900 px-1 rounded text-yellow-300">speak</code> tool.
              </p>
            )}
          </div>

          {isRoboticsER && (
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-1">TTS Model (Robotics ER Speech)</label>
              <select
                value={config.ttsModelName === "gemini-2.5-flash-preview"
                  ? "gemini-2.5-flash-preview-tts"
                  : config.ttsModelName || TTS_MODEL_OPTIONS[0]}
                onChange={(e) => setConfig({...config, ttsModelName: e.target.value})}
                className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-white"
              >
                {TTS_MODEL_OPTIONS.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Voice</label>
            <select
              value={config.voiceName}
              onChange={(e) => setConfig({...config, voiceName: e.target.value})}
              className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-white"
            >
              {VOICE_OPTIONS.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Extra Instructions</label>
            <p className="text-xs text-gray-500 mb-1">Pupper's personality and rules come from the robot (lab/prompt.md). Anything here is added after them.</p>
            <textarea
              value={config.extraInstructions}
              onChange={(e) => setConfig({...config, extraInstructions: e.target.value})}
              placeholder="Optional: try out an extra instruction without editing prompt.md..."
              rows={4}
              className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-white text-xs focus:border-cyan-500 outline-none resize-y"
            />
          </div>

          {!isRoboticsER && (
            <div className="border-t border-gray-700 pt-3">
               <label className="block text-sm font-medium text-cyan-400 mb-2">Capabilities</label>

               <label className="flex items-center space-x-2 mb-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={config.enableAffectiveDialog}
                    onChange={(e) => setConfig({...config, enableAffectiveDialog: e.target.checked})}
                    className="w-4 h-4 rounded bg-gray-700 border-gray-600 focus:ring-offset-gray-800 focus:ring-cyan-500"
                  />
                  <span className="text-sm text-gray-300">Enable Affective Dialog</span>
               </label>

               <label className="flex items-center space-x-2 mb-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={config.enableProactiveAudio}
                    onChange={(e) => setConfig({...config, enableProactiveAudio: e.target.checked})}
                    className="w-4 h-4 rounded bg-gray-700 border-gray-600 focus:ring-offset-gray-800 focus:ring-cyan-500"
                  />
                  <span className="text-sm text-gray-300">Enable Proactive Audio</span>
               </label>
            </div>
          )}

          <div className="border-t border-gray-700 pt-3">
             <label className="block text-sm font-medium text-cyan-400 mb-2">Vision & Tracking</label>

             <label className="flex items-center space-x-2 mb-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={config.enableVideo}
                  onChange={(e) => setConfig({...config, enableVideo: e.target.checked})}
                  className="w-4 h-4 rounded bg-gray-700 border-gray-600 focus:ring-offset-gray-800 focus:ring-cyan-500"
                />
                <span className="text-sm text-gray-300">Enable Vision (Send Video)</span>
             </label>

             <div className="mb-2">
               <label className="block text-xs font-medium text-gray-500 mb-1">Camera Source</label>
               <select
                  value={config.cameraSource}
                  onChange={(e) => setConfig({...config, cameraSource: e.target.value as CameraSource})}
                  className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-xs text-white"
                >
                  <option value="browser">Browser Camera (Webcam)</option>
                  <option value="server">Robot Camera (Local Server)</option>
                </select>
             </div>

             <div>
               <label className="block text-xs font-medium text-gray-500 mb-1">Eye Tracking Source</label>
               <select
                  value={config.eyeTrackingSource}
                  onChange={(e) => setConfig({...config, eyeTrackingSource: e.target.value as EyeTrackingSource})}
                  className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-xs text-white"
                >
                  <option value="server">Robot Detection (Server)</option>
                  <option value="mediapipe">Browser Detection (MediaPipe)</option>
                  <option value="mouse">Mouse (Cursor)</option>
                </select>
             </div>
          </div>

          <div className="border-t border-gray-700 pt-3">
             <label className="block text-sm font-medium text-cyan-400 mb-2">Hackathon: senses &amp; accessibility</label>

             <label className="flex items-center space-x-2 mb-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={config.captions}
                  onChange={(e) => setConfig({...config, captions: e.target.checked})}
                  className="w-4 h-4 rounded bg-gray-700 border-gray-600 focus:ring-offset-gray-800 focus:ring-cyan-500"
                />
                <span className="text-sm text-gray-300">Captions (show what Pupper says on the face)</span>
             </label>

             <label className="flex items-center space-x-2 mb-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={config.soundAlerts}
                  onChange={(e) => setConfig({...config, soundAlerts: e.target.checked})}
                  className="w-4 h-4 rounded bg-gray-700 border-gray-600 focus:ring-offset-gray-800 focus:ring-cyan-500"
                />
                <span className="text-sm text-gray-300">Sound alerts (flash LOUD SOUND on the screen)</span>
             </label>

             <label className="flex items-center space-x-2 mb-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={config.objectDetection}
                  onChange={(e) => setConfig({...config, objectDetection: e.target.checked})}
                  className="w-4 h-4 rounded bg-gray-700 border-gray-600 focus:ring-offset-gray-800 focus:ring-cyan-500"
                />
                <span className="text-sm text-gray-300">On-device object detection (obstacles, find)</span>
             </label>
             <p className="text-[11px] text-gray-500">The robot decides what to do with sounds and detections (lab/hackathon_*.py); say "stop listening" to turn sound alerts off by voice. Memory is reset from the debug view.</p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Robot API URL</label>
            <input
              type="text"
              value={config.robotApiUrl}
              onChange={(e) => setConfig({...config, robotApiUrl: e.target.value})}
              className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-white text-xs"
            />
          </div>
        </div>

        <div className="mt-6 flex items-center justify-between gap-4">
          <p className="text-xs text-gray-500">
            Gemini Pupper by{' '}
            <a href="https://scholar.google.com/citations?user=460NWeQAAAAJ&hl=en" target="_blank" rel="noreferrer" className="text-cyan-500 hover:underline">Peng Xu</a>
            {' '}(Google DeepMind), adapted for Stanford CS 123.
          </p>
          <button
            onClick={onClose}
            className="bg-cyan-600 hover:bg-cyan-700 text-white px-4 py-2 rounded font-medium transition-colors"
          >
            Save & Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;