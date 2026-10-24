# Gemini Pupper Architecture

This page explains how the browser application, Gemini models, and the physical Pupper robot work together.

## The 30-second explanation

Gemini Pupper is a browser-based voice and vision interface for a robot dog. The browser captures microphone and camera data and streams it directly to Gemini Live. When Gemini decides the robot should move or perform a trick, it returns a structured function call. The browser translates that call into an HTTP request to a small FastAPI server running locally on the robot. That server uses the Pupper ROS stack to queue motion, animation, and follow-person commands.

The cloud deployment serves the web application; it does not proxy robot commands or media. The browser is the coordinator between Gemini and the robot.

## System at a glance

```mermaid
flowchart LR
    User[Person] -->|voice, gestures, requests| Browser

    subgraph Cloud["Public cloud"]
        CloudRun["Cloud Run<br/>nginx + static React app"]
        GeminiLive["Gemini Live API<br/>audio, video, reasoning"]
        GeminiTTS["Gemini Flash TTS<br/>text-output models only"]
    end

    subgraph Client["Browser"]
        Browser["React application"]
        Media["Microphone + camera"]
        Face["Animated face + debug UI"]
        LiveService["GeminiLiveService"]
        Bridge["RobotBridge"]
    end

    subgraph Robot["Pupper on the local network"]
        API["robot/server.py<br/>FastAPI, port 8000"]
        Agent["lab/<br/>Gemini's tools"]
        Tools["robot_api.py<br/>queue, controllers"]
        ROS["Pupper ROS services"]
        Hardware["Motors, camera, battery,<br/>speaker, person tracking"]
    end

    CloudRun -->|downloads HTML, JS, assets| Browser
    Media --> LiveService
    LiveService <-->|Live WebSocket<br/>audio, video, tool calls| GeminiLive
    GeminiLive -->|native audio| LiveService
    LiveService -->|text to speech request| GeminiTTS
    GeminiTTS -->|streamed audio| LiveService
    LiveService --> Face

    LiveService -->|robot tool action| Bridge
    Face -->|manual debug action| Bridge
    Bridge <-->|HTTP JSON + JPEG<br/>local network| API
    Browser <-->|person detections<br/>WebSocket| API
    API --> Agent --> Tools --> ROS --> Hardware
    Hardware -->|camera, status, detections| API
```

### Responsibility boundaries

| Area | Responsibility | Important code |
|---|---|---|
| Cloud Run | Builds and serves the static browser application through nginx | [Dockerfile](../web/Dockerfile), [nginx.conf](../web/nginx.conf) |
| React application | Owns UI state, settings, logs, connection state, and periodic status polling | [App.tsx](../web/App.tsx) |
| Gemini integration | Captures media, opens the Live session, plays audio, executes tool calls, and returns tool results | [web/services/geminiLive.ts](../web/services/geminiLive.ts) |
| Robot bridge | Maps browser operations to the robot's HTTP API | [web/services/robotBridge.ts](../web/services/robotBridge.ts) |
| Robot API | Exposes control, camera, tracking, and telemetry endpoints on port 8000 | [robot/server.py](../robot/server.py) |
| Gemini's tools | The prompt, what Gemini reads about each tool, and what each tool does | [lab/prompt.md](../lab/prompt.md) and the `@tool`s and behaviors in [lab/](../lab/) |
| Robot side | The job queue, the robot's state, leg-controller switching, interactive tricks, falls, follow mode, the e-stop | [robot/robot_api.py](../robot/robot_api.py) |
| Robot stack | Converts high-level requests into queued ROS motion and behavior operations | [Pupper v3 monorepo](https://github.com/Nate711/pupperv3-monorepo) |

## End-to-end conversation and action

The browser, rather than Cloud Run, owns the live conversation and function-call loop.

```mermaid
sequenceDiagram
    autonumber
    actor Person
    participant Browser as Browser app
    participant Gemini as Gemini Live
    participant RobotAPI as Robot FastAPI
    participant ROS as lab/ + robot_api.py
    participant Pupper as Robot hardware

    Person->>Browser: Speaks and appears on camera
    Browser->>Gemini: Streams PCM audio and JPEG frames
    Gemini-->>Browser: Native audio response
    Browser-->>Person: Plays response and animates mouth

    Gemini-->>Browser: Function call: play_trick or move
    Browser->>RobotAPI: POST /tools/play_trick or /tools/move
    RobotAPI->>ROS: Check the request, queue a job
    ROS-->>RobotAPI: Result: accepted (queued) or refused, with a reason
    RobotAPI-->>Browser: JSON result
    Browser->>Gemini: Tool response
    ROS->>Pupper: Later, when the queue gets there: switch to the walking or trick policy
    Pupper-->>ROS: Outcome, in the debug log and robot_status
```

A manual button in the debug view enters the same path at the browser-to-robot HTTP step, bypassing Gemini. This makes it useful for checking the robot connection and motion APIs independently.

## Audio and model modes

The application supports two response patterns:

### Native-audio Live models

Gemini 3.1 Flash Live and Gemini 2.5 Flash Native Audio return audio directly through the Live session.

```text
microphone -> browser -> Gemini Live -> native audio -> browser speakers
```

The separate `speak` tool is intentionally not advertised to native-audio models.

### Text-output robotics model

The Robotics ER streaming model returns text and can call the `speak` tool. The browser sends that text to the selected Gemini Flash TTS model and plays the returned audio.

```text
robotics model text -> speak tool -> Gemini Flash TTS -> browser speakers
```

Gemini 3.1 Flash TTS is consumed as a stream, so playback can begin before the entire utterance has been generated. Gemini 2.5 Flash TTS uses a complete-response request because that model does not support TTS streaming.

## Robot-side architecture

The robot server is a translation and observation layer between browser-friendly protocols and the Pupper ROS environment.

```mermaid
flowchart TB
    subgraph Inputs["Inputs to robot/server.py"]
        REST["REST commands<br/>movement, animation, follow, stop"]
        Poll["REST polling<br/>status and camera"]
        WSClient["Browser WebSocket<br/>/ws/detections"]
    end

    subgraph Server["FastAPI process on port 8000"]
        Routes["FastAPI routes"]
        Agent["tool_registry.py + lab/ tools<br/>checks each call, words to numbers"]
        Status["Status collectors<br/>battery, CPU, services, network"]
        Camera["Latest ROS JPEG queue"]
        Broadcast["Detection broadcaster"]
    end

    subgraph RobotStack["Robot processes"]
        ToolServer["lab/ behaviors + robot_api.py<br/>check now, queue a job"]
        Motion["Motion and animation queues"]
        Follow["Person-follow behavior"]
        CameraROS["ROS camera"]
        Tracker["Person detector<br/>ZMQ tcp://127.0.0.1:5556"]
    end

    REST --> Routes --> Agent --> ToolServer
    ToolServer --> Motion
    ToolServer --> Follow

    Poll --> Status
    Poll --> Camera
    CameraROS --> Camera

    Tracker -->|JSON detections| Broadcast --> WSClient
```

### Robot command API

| Purpose | Method and path | Browser caller |
|---|---|---|
| Gemini's prompt and tools | `GET /tools` (lab/prompt.md, the tools in lab/) | Session setup |
| Run one of Gemini's tools | `POST /tools/<name>` | Every Gemini tool call |
| Activate or relax motors | `POST /walking/activate`, `POST /walking/deactivate` | Wake/sleep tools |
| Move with velocities | `POST /move/velocity` | Debug controls |
| Move by heading | `POST /move/direction` | Directional debug control |
| Stop or clear work | `POST /queue/stop`, `POST /queue/reset`, `POST /stop/immediate` | Safety and debug controls |
| Play a trick | `POST /animation` | Debug controls |
| Follow a person | `POST /following/activate`, `POST /following/deactivate` | Follow-mode tool |
| Read camera | `GET /camera/image` | Debug view and optional Gemini vision source |
| Read health | `GET /system/status` | Face warnings and debug status bar |
| Track a person | `WS /ws/detections` | Animated eye direction |
| Set speaker volume | `POST /system/volume` | Debug control |
| Due reminders to tell the person | `POST /reminders/claim`, `POST /reminders/<id>/delivered` | Reminder poller (every 5 s); see README, Reminders |
| List reminders | `GET /reminders` | Debugging (`curl`) |

The animation endpoint plays a trick from `tricks/` by id, such as `sneeze` or `swim`. Every trick is played by policies, so playing one switches the legs to that trick's controller (or, for the jump, feeds a reference to the locomotion policy), and the command queue waits for the trick to finish so later actions do not overlap it. The robot is always in exactly one state: locomotion, one trick (trick mode), motors idle, or fallen.

### Camera and person tracking

There are two distinct visual paths:

1. **Vision sent to Gemini:** The app can use the browser camera or poll `/camera/image` from the robot. Frames are resized, JPEG-compressed, and sent to the Live session.
2. **Eye animation:** The face can track the mouse, run MediaPipe in the browser, or consume person detections from the robot. In robot mode, a local detector publishes JSON over ZMQ; the FastAPI process forwards it to the browser over `/ws/detections`.

These paths are independent: changing the animated-eye tracking source does not necessarily change which camera Gemini sees.

### Telemetry

The robot API assembles status from several local sources:

- Battery percentage and voltage from the Pupper battery utility
- CPU utilization and temperature through `psutil`
- Robot service state through `systemctl is-active robot`
- Internet reachability through a short socket connection test

The main application polls slowly for critical warnings, while the debug panel polls more frequently for operator feedback.

## Deployment and network topology

```mermaid
flowchart TB
    Source["GitHub repository"] -->|gcloud run deploy --source| Build["Google Cloud Build"]
    Build --> Image["Container image<br/>Vite build + nginx"]
    Image --> Run["Public Cloud Run service<br/>port 8080"]
    Run -->|HTTPS static assets| Browser["Chrome browser"]

    Browser -->|Google API connection| Gemini["Gemini APIs"]
    Browser -->|http://localhost:8000<br/>or robot LAN address| Robot["Pupper FastAPI"]
```

The Cloud Run container contains only static files and nginx. There is no application backend in Cloud Run and robot traffic does not pass through Google Cloud.

The default robot URL is `http://localhost:8000`. This works when Chrome runs on the Pupper itself. If Chrome runs on another computer, set the Robot API URL to the Pupper's reachable LAN address and ensure browser mixed-content and local-network permission policies allow that connection.

## Trust and safety boundaries

- The Gemini API key is entered in the browser settings and stored in browser local storage. It is used for direct browser-to-Gemini connections.
- The robot FastAPI server currently permits all CORS origins and does not authenticate control requests.
- Keep port 8000 on a trusted robot/local network. Do not expose it directly to the public internet.
- Motion commands can move physical hardware. Use the immediate-stop path when behavior is unexpected, and keep physical clearance around the robot.
- The server includes maintenance endpoints such as upgrade and restart. Network isolation is therefore important even if the browser UI does not normally invoke them.

## Startup lifecycle

1. `robot/start.sh` launches the robot stack (`robot/launch/gemini_pupper.launch.py`: every leg controller loaded, none active) and then the FastAPI process (`robot/server.py`).
2. FastAPI's lifespan hook creates the `Robot` (`robot/robot_api.py`); its queue first checks which state the robot was left in before a restart (an interactive trick it is still in, say).
3. If ZMQ is available, a background subscriber connects to the local person-detection publisher.
4. The user opens the Cloud Run URL in Chrome.
5. The browser loads settings and connects to the robot API.
6. When the user wakes Pupper, the browser requests media access and opens the Gemini Live session.
7. Gemini responses, robot commands, status polling, and tracking streams then operate independently until disconnect.

## Failure isolation

| Symptom | Likely boundary to inspect |
|---|---|
| Page does not load | Cloud Run, nginx, browser cache |
| Camera or microphone unavailable | Browser permissions, HTTPS secure context, Linux audio/video devices |
| Gemini connects and immediately closes | Model ID, API key, unsupported Live configuration; inspect close code in debug logs |
| Voice works but robot does not move | Robot API URL, LAN access, CORS/mixed-content policy, the robot server (`/tmp/gemini-pupper/server.log`) |
| Manual controls work but Gemini actions do not | Gemini function declarations and tool-call handling |
| Robot moves but eye tracking is absent | ZMQ detector, port 5556 publisher, `/ws/detections` |
| Debug status is empty | `/system/status`, robot service, battery script, network reachability |

## Related documentation

- [Project overview and setup](interactive-robot-dog.md)
- [Robot-side server](../robot/server.py), [the lab: tools and behaviors](../lab/), [the Robot API](../robot/robot_api.py)
- [Repository README](../README.md)
- [Pupper v3 robot stack](https://github.com/Nate711/pupperv3-monorepo)
