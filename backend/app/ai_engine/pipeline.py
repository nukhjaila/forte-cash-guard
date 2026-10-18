"""Entry point used by backend/app/main.py: detector + rules (see rules_engine.py)."""
from backend.app.ai_engine.rules_engine import analyze_video


def analyze_video_with_qwen(video_path: str):
    # name kept so main.py does not change; the video analysis no longer uses a vision language model
    return analyze_video(video_path, use_cache=False, with_frames=True)