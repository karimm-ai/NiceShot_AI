import argparse
import sys
import json
from niceshot_ai.niceshot_ai import NiceShot_AI

def main():
    parser = argparse.ArgumentParser(description="NiceShot AI CLI")

    # Required
    parser.add_argument("--game", required=True, help="Name of the game")
    parser.add_argument("--input", required=True, help="Path to video file")
    parser.add_argument("--output", required=True, help="Output directory")

    # Optional

    parser.add_argument("--save_clips", action="store_true")
    parser.add_argument("--vertical_format", action="store_true")
    parser.add_argument("--session_analysis", action="store_true")
    parser.add_argument("--compilation", action="store_true")
    parser.add_argument("--comp_len", type=int, default=0)
    parser.add_argument("--coaching", default="None")

    args = parser.parse_args()

    try:
        #check_and_update()
        
        niceshot_ai = NiceShot_AI(
            args.game,
            args.input,
            total_hours=10000,
            save_clips=args.save_clips,
            output_dir=args.output,
            max_workers=2,
            frame_idx_start=0,
            frames_to_skip=8,
            add_to_csv=True,
            create_montage=args.compilation,
            montage_length_sec=args.comp_len,
            max_videos=1,
            vertical_format=args.vertical_format,
            advanced_detection=True,
            session_analysis=args.session_analysis,
            coaching=args.coaching
        )

        niceshot_ai.run()

        with open(f"{args.output}/status.json", "w") as f:
            json.dump({"status": "completed", "error": None}, f)

        sys.exit(0)

    except Exception as e:
        with open(f"{args.output}/status.json", "w") as f:
            json.dump({"status": "failed", "error": str(e)}, f)

        sys.exit(1)

if __name__ == "__main__":
    main()