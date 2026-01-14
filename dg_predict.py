#!/usr/bin/env python3
"""DG Predict helper script for inference with exported models. #DG"""

import argparse

from ultralytics import YOLO


def parser_arguments():
    parser = argparse.ArgumentParser(description="Run YOLO prediction")
    parser.add_argument("--weights", type=str, required=True, help="model weights path")
    parser.add_argument("--source", type=str, required=True, help="image/video source path")
    parser.add_argument("--device", type=str, default="0", help="device to use")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="image size (pixels)")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--save", action="store_true", default=True, help="save results")
    parser.add_argument("--no-separate-outputs", action="store_true", help="exported file without separate outputs")

    return parser.parse_args()


if __name__ == "__main__":
    args = parser_arguments()
    kwargs = vars(args)
    print(f"Predict args: {kwargs}")

    model = YOLO(args.weights)

    # Auto-detect separate_outputs based on file extension
    separate_outputs = not args.no_separate_outputs and not args.weights.endswith(".pt")

    results = model.predict(
        args.source,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        save=args.save,
        separate_outputs=separate_outputs,
    )

    print(f"Prediction completed. Results: {len(results)} images processed.")
