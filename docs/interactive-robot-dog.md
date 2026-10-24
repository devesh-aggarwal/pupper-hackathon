# Interactive Robot Dog with Gemini Live

Contact: [Peng Xu](mailto:sippey@gmail.com) · [@sippeyxp](https://x.com/sippeyxp)

Source: [Google Doc](https://docs.google.com/document/d/1Evuc-yfxR6ow5dA9LsWJpZFUy-mzFpg1HyeJsRQg9jc/edit?tab=t.0)

## Contents

- [What is it?](#what-is-it)
- [System architecture](architecture.md)
- [How to get a Pupper](#how-to-get-a-pupper)
- [Robot setup](#robot-setup)
- [Development](#development)

## What is it?

This project connects a physical Pupper robot dog to Gemini Live. The browser app uses a microphone and camera for real-time interaction, and Gemini function calls trigger supported robot movements and animations.

### Real-robot videos

- [Function demo](https://youtu.be/ZowEWyZ4geg)
- [Demonstration of robot capabilities](https://youtube.com/shorts/EfL2XWKZX8E)

Additional browser demonstrations include:

- Play a new game
- Teach a new trick
- Show capabilities
- Talk in an accent

You can [try the browser app](https://gemini-pupper-746329146697.us-central1.run.app) without a robot.

## How to get a Pupper

Pupper is an open-source project. You can build one from the project designs, or purchase a kit or assembled Pupper from [Present Perfection](https://www.present-perfection.com/).

## Robot setup

> These commands change services and application code on the robot. Run them on the Pupper over SSH, not on your development computer.

1. Install the AI-version system image on the Pupper.
2. Connect to the robot over SSH.
3. Download or copy [`robot/local_server.py`](../robot/local_server.py) onto the robot.
4. Stop the existing agent and UI services:

   ```bash
   sudo systemctl stop llm-agent
   sudo systemctl stop pupper-ui
   ```

5. Back up the existing agent, then replace it with the local robot server:

   ```bash
   cd /home/pi/pupperv3-monorepo/ai/llm-ui/agent-starter-python/src
   cp agent.py agent.py.backup
   cp /path/to/local_server.py agent.py
   ```

6. Restart the agent service:

   ```bash
   sudo systemctl restart llm-agent
   ```

7. Open Chrome and navigate to the [Gemini Pupper browser app](https://gemini-pupper-746329146697.us-central1.run.app).

The robot-side server expects the Pupper software environment, including `ros_tool_server`, FastAPI, Uvicorn, and the robot's ROS services.

## Development

### Front end

You can begin by making a copy of the [AI Studio app](https://ai.studio/apps/drive/1N-3Kw5AEe9GuSUjazvRzaiRt2_K3DpXf).

See the [development walkthrough video](https://www.youtube.com/watch?v=kgKC4wosUEw).

### Prompt used in the walkthrough

> I would like to make a robot dog app. The app uses the Gemini Live model to interact with the user through a microphone and camera. If the user's request matches an action the robot dog can perform, the model makes a function call to invoke that action.
>
> The robot dog can walk, turn, follow people, and perform predefined trick actions. The detailed list of possible motions is in the attached Python server, which the front end calls to move the robot.
>
> The UI has two alternating panes: face and debug.
>
> 1. The face pane shows a dog face. Its eyes track mouse movement, and its mouth matches audio output from the Live model.
> 2. The debug pane contains a camera view in the upper left, a log window in the lower left, and debugging buttons on the right that invoke functions directly. It also displays system status.
>
> The app should also include a settings page for all options.

### Back end and robot stack

See the [Pupper v3 monorepo](https://github.com/Nate711/pupperv3-monorepo) and its documentation for the underlying robot stack.
