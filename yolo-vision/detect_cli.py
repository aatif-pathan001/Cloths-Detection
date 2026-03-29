"""
Command-line interface for YOLO Vision.

Examples:
    python detect_cli.py image --model yolov8n.pt --input photo.jpg
    python detect_cli.py video --model yolov8n.pt --input clip.mp4 --output out.mp4
    python detect_cli.py webcam --model yolov8n.pt --camera 0
"""

import argparse
import sys
import cv2
from pathlib import Path

from app.core.detector import YOLODetector
from app.utils.helpers import ensure_output_dirs, build_output_path, summarize_detections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yolo-vision",
        description="YOLO Vision — CLI Object Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True, help="Path to YOLO .pt model file")
    common.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    common.add_argument("--iou", type=float, default=0.45, help="IOU threshold")

    # Image subcommand
    img = sub.add_parser("image", parents=[common], help="Detect objects in an image")
    img.add_argument("--input", required=True, help="Path to input image")
    img.add_argument("--output-dir", default="outputs/images", help="Directory to save results")
    img.add_argument("--show", action="store_true", help="Display result in a window")

    # Video subcommand
    vid = sub.add_parser("video", parents=[common], help="Process a video file")
    vid.add_argument("--input", required=True, help="Path to input video")
    vid.add_argument("--output", default=None, help="Output video path")

    # Webcam subcommand
    cam = sub.add_parser("webcam", parents=[common], help="Live webcam detection")
    cam.add_argument("--camera", type=int, default=0, help="Camera device index")

    return parser


def cmd_image(args, detector: YOLODetector):
    """Run image detection from CLI."""
    print(f"  Input  : {args.input}")
    annotated, detections = detector.detect_image(args.input)

    out_dirs = ensure_output_dirs("outputs")
    out_path = build_output_path(args.input, args.output_dir)
    cv2.imwrite(out_path, annotated)
    crops = detector.save_crops(detections, str(out_dirs["crops"]))

    summary = summarize_detections(detections)
    print(f"  Objects: {summary['total_objects']}  |  Classes: {list(summary['class_counts'].keys())}")
    print(f"  Saved  : {out_path}")
    print(f"  Crops  : {len(crops)} files → outputs/crops/")

    if args.show:
        cv2.imshow("YOLO Vision", annotated)
        print("  Press any key to close…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cmd_video(args, detector: YOLODetector):
    """Run video detection from CLI."""
    out_path = args.output or build_output_path(args.input, "outputs/videos", "_detected")
    print(f"  Input  : {args.input}")
    print(f"  Output : {out_path}")

    frame_counter = [0]

    def progress(current, total):
        pct = int(current / max(total, 1) * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  Frame {current}/{total}", end="", flush=True)
        frame_counter[0] = current

    stats = detector.detect_video(args.input, out_path, progress_callback=progress)
    print(f"\n  Done   : {stats['frames_processed']} frames · {stats['total_detections']} detections")
    print(f"  Saved  : {out_path}")


def cmd_webcam(args, detector: YOLODetector):
    """Run live webcam detection from CLI (press Q to quit)."""
    print(f"  Camera : index {args.camera}")
    print("  Press Q in the window to quit.")

    for annotated_frame, detections in detector.stream_webcam(args.camera):
        summary = summarize_detections(detections)
        cv2.putText(
            annotated_frame,
            f"Objects: {summary['total_objects']}  Conf: {summary['avg_confidence']:.0%}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 212, 255), 2, cv2.LINE_AA,
        )
        cv2.imshow("YOLO Vision — Webcam  (Q to quit)", annotated_frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyAllWindows()


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("\n══ YOLO Vision CLI ══════════════════════")
    print(f"  Model  : {args.model}")
    print(f"  Conf   : {args.conf}  |  IOU: {args.iou}")

    detector = YOLODetector(args.model, confidence_threshold=args.conf, iou_threshold=args.iou)
    detector.load_model()
    print(f"  Loaded : {len(detector.class_names)} classes")
    print("────────────────────────────────────────\n")

    if args.mode == "image":
        cmd_image(args, detector)
    elif args.mode == "video":
        cmd_video(args, detector)
    elif args.mode == "webcam":
        cmd_webcam(args, detector)

    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
