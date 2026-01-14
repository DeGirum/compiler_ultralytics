#!/usr/bin/env python3
"""DG Validation helper script for model evaluation. #DG"""

import argparse

from ultralytics import YOLO


def parser_arguments():
    parser = argparse.ArgumentParser(description="Validate YOLO model")
    parser.add_argument("--weights", type=str, required=True, help="model weights path")
    parser.add_argument("--data", type=str, default="coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--annotations", type=str, default=None, help="ground truth annotation JSON file path")
    parser.add_argument("--device", type=str, default="cpu", help="device to use")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="image size (pixels)")
    parser.add_argument("--no-separate-outputs", action="store_true", help="exported file without separate outputs")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--conf", type=float, default=0.001, help="confidence threshold")

    return parser.parse_args()


if __name__ == "__main__":
    args = parser_arguments()
    kwargs = vars(args)
    print(f"Validation args: {kwargs}")

    model = YOLO(args.weights)

    # Auto-detect separate_outputs based on file extension
    separate_outputs = not args.no_separate_outputs and not args.weights.endswith(".pt")
    save_json = True if args.annotations else False

    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        conf=args.conf,
        separate_outputs=separate_outputs,
        save_json=save_json,
        anno_json=args.annotations,
    )

    print(f"Validation completed.")
    print(f"Results: {results.results_dict}")
