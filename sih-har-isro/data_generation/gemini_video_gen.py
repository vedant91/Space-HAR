"""
Gemini Veo 2 Synthetic Dataset Generator
=========================================
Generates synthetic training videos for each experiment step using
Google's Gemini Veo 2 video generation model.

Usage:
    python data_generation/gemini_video_gen.py --step all --count 5

Requirements:
    pip install google-genai opencv-python
    Set GEMINI_API_KEY environment variable.
"""

import os
import sys
import time
import argparse
import logging
import urllib.request
from pathlib import Path
from typing import Optional

# Ensure project root is in path (works when run directly OR as module)
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

# Attempt Gemini import (gracefully fail if not installed)
try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("[WARNING] google-genai not installed. Run: pip install google-genai")

from config.experiment_config import (
    EXPERIMENT_STEPS,
    GEMINI_MODEL,
    SYNTHETIC_OUTPUT_DIR,
    FRAMES_EXTRACT_PER_VIDEO,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Space Environment Base Context ────────────────────────────────────────────
# Injected into every prompt to establish the ISS / spacecraft setting.
SPACE_CONTEXT = (
    "Inside a spacecraft science module resembling the International Space Station interior. "
    "The astronaut wears a full white EVA-style pressure suit with a clear visor helmet, "
    "thick pressurized gloves, and mission patches visible on the sleeves. "
    "The walls are covered in beige-grey paneling with cable bundles, handrails, and equipment racks. "
    "Fluorescent strip lighting with a cool white tone. "
    "A payload experiment rack is mounted to the module wall with the experiment box secured via "
    "Velcro and tethers. "
    "A fixed payload camera is mounted above the experiment rack at a 45-degree downward angle. "
    "Microgravity environment — small floating particles are visible in the air, "
    "and the astronaut braces against a handrail with one foot. "
    "The motion is slow and deliberate as required in zero-gravity conditions. "
    "Cinematic, photorealistic, 4K quality video."
)

# ── Prompt Templates per Step ──────────────────────────────────────────────────
STEP_PROMPTS = {
    1: [
        f"{SPACE_CONTEXT} "
        "The astronaut slowly moves both gloved hands toward a white rectangular experiment "
        "container secured to the payload rack. The helmet visor reflects the experiment lighting. "
        "Fixed overhead payload camera at 45 degrees. The astronaut positions their thick gloved "
        "hands near the latched white container, ready to begin. Duration: 8 seconds.",

        f"{SPACE_CONTEXT} "
        "Side-angle fixed camera inside the ISS module. The astronaut floats gently and grips a "
        "handrail with one hand while moving the other gloved hand toward a white storage box "
        "mounted on the experiment rack. Slow, weightless movement. Duration: 8 seconds.",
    ],
    2: [
        f"{SPACE_CONTEXT} "
        "Overhead fixed payload camera. The astronaut uses both thick pressurized gloves to "
        "carefully unlatch and open the lid of a white rectangular experiment container mounted "
        "to the rack. Inside the container, a small red box and a small yellow box are visible, "
        "secured with foam padding. The lid opens slowly in microgravity. Duration: 6 seconds.",

        f"{SPACE_CONTEXT} "
        "Front-facing fixed camera. The astronaut's gloved hands grip the sides of a hinged "
        "white container and push the lid open with deliberate force. The red and yellow inner "
        "boxes are revealed inside. Helmet visor reflects the interior lighting. Duration: 6 seconds.",
    ],
    3: [
        f"{SPACE_CONTEXT} "
        "Overhead payload camera at 45 degrees. The astronaut reaches into the open white "
        "container with both thick white pressurized gloves and carefully grasps a small red box. "
        "The astronaut lifts the red box slowly out of the container, mindful of microgravity. "
        "The red box is clearly visible against the white suit gloves. Duration: 8 seconds.",

        f"{SPACE_CONTEXT} "
        "Close-up 45-degree angle fixed camera. Two white pressurized astronaut gloves grasp "
        "and lift a small red rectangular box from a larger white experiment container mounted "
        "to the ISS payload rack. The movement is slow and deliberate, typical of zero-gravity "
        "object handling. Tethers visible on the box. Duration: 8 seconds.",
    ],
    4: [
        f"{SPACE_CONTEXT} "
        "Fixed front camera. The astronaut holds a red box in both gloved hands, rotating it "
        "slowly 90 degrees while examining every surface. The helmet faces down toward the box. "
        "The red box floats slightly in the astronaut's hands. ISS module background visible. "
        "Duration: 8 seconds.",

        f"{SPACE_CONTEXT} "
        "Side-angle camera inside ISS. The astronaut examines a small red box held in white "
        "pressurized gloves, tilting and turning it carefully. The astronaut's helmet visor "
        "reflects the box. Slow, methodical movement characteristic of microgravity operations. "
        "Duration: 8 seconds.",
    ],
    5: [
        f"{SPACE_CONTEXT} "
        "Overhead payload camera. The astronaut uses both gloved hands to carefully place the "
        "red box into a designated marked zone on the LEFT side of the experiment rack surface. "
        "The box is gently lowered and secured. The marked zone is labeled with a red indicator. "
        "Deliberate, precise placement in microgravity. Duration: 6 seconds.",

        f"{SPACE_CONTEXT} "
        "Front camera view of the payload rack. The astronaut's white gloved hands gently lower "
        "a red box onto a Velcro-marked designated area to the left of the main container. "
        "The box adheres to the surface. Careful, precise motion. Duration: 6 seconds.",
    ],
    6: [
        f"{SPACE_CONTEXT} "
        "Overhead payload camera at 45 degrees. The astronaut reaches back into the open white "
        "container with both gloved hands and picks up the yellow box remaining inside. "
        "The yellow box is lifted slowly and clearly out of the white container. Microgravity "
        "motion — the astronaut stabilizes with one foot on a restraint bar. Duration: 8 seconds.",

        f"{SPACE_CONTEXT} "
        "45-degree fixed camera. Two white pressurized astronaut gloves reach into the "
        "experiment container and grasp a small yellow rectangular box. The yellow color "
        "contrasts sharply with the white suit and white container. Slow lift. Duration: 8 seconds.",
    ],
    7: [
        f"{SPACE_CONTEXT} "
        "Fixed front camera inside ISS module. The astronaut holds a yellow box in both "
        "pressurized gloves and examines it from multiple angles, rotating it slowly. "
        "The yellow box is clearly visible against the white suit. Helmet faces the box. "
        "ISS equipment racks visible in background. Duration: 8 seconds.",

        f"{SPACE_CONTEXT} "
        "Side-view camera. The astronaut examines a small yellow box held in white gloved hands, "
        "carefully inspecting all surfaces. The motion is slow and methodical. "
        "ISS module interior with cable bundles and panels in background. Duration: 8 seconds.",
    ],
    8: [
        f"{SPACE_CONTEXT} "
        "Overhead payload camera. The astronaut carefully places the yellow box into the "
        "designated marked zone on the RIGHT side of the experiment rack surface. "
        "Both gloved hands lower the box precisely. Yellow indicator marking visible on the surface. "
        "The red box is already in its left zone. Duration: 6 seconds.",

        f"{SPACE_CONTEXT} "
        "Front camera of payload rack. The astronaut's gloved hands lower a yellow box onto "
        "a Velcro-marked area to the right of the main container. The red box is already "
        "placed on the left. Both boxes now visible in their zones. Duration: 6 seconds.",
    ],
}

# ── Orientation Variants for Microgravity Training ─────────────────────────────
# These suffixes are appended to prompts to generate orientation-agnostic training data.
ORIENTATION_VARIANTS = [
    # Normal orientation
    "The experiment rack is mounted vertically on the ISS module wall. "
    "Astronaut is upright relative to the rack.",

    # Rotated ~90 degrees (astronaut upside down relative to rack)
    "The experiment rack is mounted on the ISS module CEILING. "
    "The astronaut is working upside-down relative to the Earth frame, "
    "floating horizontally with feet toward the ceiling. "
    "This represents a typical microgravity orientation in space.",

    # Rotated ~180 degrees
    "The experiment rack is mounted on the ISS module FLOOR. "
    "The astronaut floats above the rack, working downward. "
    "All handrails are below the workspace. "
    "Microgravity — the astronaut has no preferred 'up' direction.",

    # Tilted / lateral orientation
    "The experiment rack is mounted on a SIDE WALL of the ISS module at an unusual angle. "
    "The astronaut is working sideways, rotated 90 degrees from the standard orientation. "
    "This is common in the Columbus and Kibo ISS modules.",
]


def generate_video_for_step(
    client,
    step_id: int,
    prompt: str,
    output_path: str,
) -> bool:
    """Generate a single video using Veo 3.1, with retry on 429 quota errors."""
    if not GENAI_AVAILABLE:
        logger.error("google-genai not available.")
        return False

    logger.info("Generating video for Step %d → %s", step_id, output_path)
    logger.info("Prompt: %s", prompt[:80] + "...")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            # Veo 3.1 preview: duration_seconds not supported — omit it
            operation = client.models.generate_videos(
                model=GEMINI_MODEL,
                prompt=prompt,
                config=genai_types.GenerateVideosConfig(
                    aspect_ratio="16:9",
                ),
            )

            # Poll until done
            max_wait = 360
            waited = 0
            logger.info("  Generation started (attempt %d). Waiting ~2-3 min...", attempt)
            while not operation.done:
                time.sleep(15)
                waited += 15
                operation = client.operations.get(operation)
                logger.info("  Still generating... %ds elapsed", waited)
                if waited >= max_wait:
                    logger.error("Timed out for step %d", step_id)
                    return False

            generated = operation.response.generated_videos
            if not generated:
                logger.error("No videos returned for step %d", step_id)
                return False

            video_obj = generated[0].video
            if hasattr(video_obj, "uri") and video_obj.uri:
                logger.info("Downloading from URI...")
                urllib.request.urlretrieve(video_obj.uri, output_path)
            elif hasattr(video_obj, "video_bytes") and video_obj.video_bytes:
                logger.info("Saving from bytes (%d bytes)", len(video_obj.video_bytes))
                with open(output_path, "wb") as f:
                    f.write(video_obj.video_bytes)
            else:
                logger.error("No downloadable data for step %d", step_id)
                return False

            logger.info("Saved: %s", output_path)
            return True

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait_sec = 65 * attempt
                logger.warning(
                    "  429 quota hit (attempt %d/%d). Waiting %ds...",
                    attempt, max_retries, wait_sec
                )
                time.sleep(wait_sec)
            else:
                logger.error("Error generating video for step %d: %s", step_id, e)
                return False

    logger.error("All %d attempts failed for step %d", max_retries, step_id)
    return False


def extract_frames_from_video(video_path: str, output_dir: str, step_id: int, num_frames: int = 100):
    """Extract evenly-spaced frames from a video for YOLO/LSTM training."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        logger.warning("No frames found in %s", video_path)
        return []

    indices = [int(i * total / num_frames) for i in range(num_frames)]
    saved = []

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"step{step_id:02d}_synthetic_{Path(video_path).stem}_frame{i:04d}.jpg"
        fpath = os.path.join(output_dir, fname)
        cv2.imwrite(fpath, frame)
        saved.append(fpath)

    cap.release()
    logger.info("Extracted %d frames from %s → %s", len(saved), video_path, output_dir)
    return saved


def run_generation(step_ids: Optional[list] = None, videos_per_step: int = 3):
    """Main generation loop."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Get your key from: https://aistudio.google.com/app/apikey"
        )

    if not GENAI_AVAILABLE:
        raise ImportError("Install google-genai: pip install google-genai")

    client = genai.Client(api_key=api_key)

    steps_to_process = EXPERIMENT_STEPS if step_ids is None else [
        s for s in EXPERIMENT_STEPS if s["id"] in step_ids
    ]

    video_dir = Path(SYNTHETIC_OUTPUT_DIR) / "videos"
    frame_dir = Path(SYNTHETIC_OUTPUT_DIR) / "frames"
    video_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    total_frames = 0

    for step in steps_to_process:
        sid = step["id"]
        prompts = STEP_PROMPTS.get(sid, [])

        for p_idx, prompt in enumerate(prompts[:videos_per_step]):
            vid_name = f"step{sid:02d}_v{p_idx+1:02d}.mp4"
            vid_path = str(video_dir / vid_name)

            if os.path.exists(vid_path):
                logger.info("Skipping (exists): %s", vid_path)
            else:
                # Rate limit: sleep 35s BEFORE every request (free tier ~2 RPM)
                logger.info("  Waiting 35s for rate limit before next request...")
                time.sleep(35)

                success = generate_video_for_step(client, sid, prompt, vid_path)
                if not success:
                    continue
                total_generated += 1

            # Extract frames for training
            step_frame_dir = str(frame_dir / f"step_{sid:02d}")
            frames = extract_frames_from_video(
                vid_path, step_frame_dir, sid, num_frames=FRAMES_EXTRACT_PER_VIDEO
            )
            total_frames += len(frames)

    logger.info("=" * 60)
    logger.info("Generation complete: %d videos, %d frames", total_generated, total_frames)
    logger.info("Frames saved to: %s", str(frame_dir))


def create_mock_dataset(steps_count: int = 8, frames_per_step: int = 50):
    """
    Create a MOCK dataset using webcam (no API needed).
    For development/testing when Gemini API is unavailable.
    Uses color-coded synthetic frames.
    """
    import numpy as np

    frame_dir = Path(SYNTHETIC_OUTPUT_DIR) / "frames"
    logger.info("Creating mock dataset (no API) with %d frames/step", frames_per_step)

    STEP_COLORS = {
        1: (200, 200, 200), 2: (180, 180, 220), 3: (50, 50, 200),
        4: (30, 30, 180),   5: (60, 60, 210),   6: (30, 200, 200),
        7: (20, 180, 180),  8: (40, 210, 210),
    }

    for step in EXPERIMENT_STEPS:
        sid = step["id"]
        step_dir = frame_dir / f"step_{sid:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        color = STEP_COLORS.get(sid, (128, 128, 128))

        for i in range(frames_per_step):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise + np.array(color), 0, 255).astype(np.uint8)

            # Draw text label
            cv2.putText(frame, f"Step {sid}: {step['name']}",
                        (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(frame, f"Frame {i+1}/{frames_per_step}",
                        (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

            fname = step_dir / f"step{sid:02d}_mock_frame{i:04d}.jpg"
            cv2.imwrite(str(fname), frame)

    logger.info("Mock dataset created at: %s", str(frame_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Veo 2 Synthetic Dataset Generator")
    parser.add_argument("--step", type=str, default="all",
                        help="Step IDs to generate (comma-separated) or 'all'")
    parser.add_argument("--count", type=int, default=2,
                        help="Number of videos per step (max ~3 per prompt variant)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock dataset (no API needed, for testing)")
    args = parser.parse_args()

    if args.mock:
        create_mock_dataset()
    else:
        step_ids = None if args.step == "all" else [int(x) for x in args.step.split(",")]
        run_generation(step_ids=step_ids, videos_per_step=args.count)
