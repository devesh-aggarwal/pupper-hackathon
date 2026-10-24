#!/usr/bin/env python3
"""Talk to Gemini in text, and watch every tool call it makes.

    python3 lab/chat.py            # your prompt and tools, on a fake robot: nothing moves
    python3 lab/chat.py --robot    # your tools on the real Pupper: it moves!

The chat itself is in robot/gemini_chat.py (python3 lab/chat.py --help for more).
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "robot" / "gemini_chat.py"), run_name="__main__")
