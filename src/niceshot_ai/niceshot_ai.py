import argparse
import sys
import json
from detector import EventDetector
from updater import check_and_update

def main():
    parser = argparse.ArgumentParser(description="NiceShot AI CLI")

    # Required
    parser.add_argument("--input", required=True, help="Path to video file")
    parser.add_argument("--output", required=True, help="Output directory")

    # Optional
    parser.add_argument("--game", default="Call of Duty Black Ops 6")
 
    args = parser.parse_args()

    try:
        check_and_update()
        
        detector = EventDetector(
            args.game,
            args.input,
            total_hours=100,
            save_clips=True,
            output_dir=args.output,
            max_workers=2,
            frame_idx_start=0,
            frames_to_skip=8,
            add_to_csv=True,
            create_montage=True,
            montage_length_sec=50,
            max_videos=1,
            vertical_format=False,
            advanced_detection=True,
            session_analysis=False
        )

        detector.detect_events()

        with open(f"{args.output}/status.json", "w") as f:
            json.dump({"status": "completed", "error": None}, f)

        sys.exit(0)

    except Exception as e:
        with open(f"{args.output}/status.json", "w") as f:
            json.dump({"status": "failed", "error": str(e)}, f)

        sys.exit(1)


if __name__ == "__main__":
    main()