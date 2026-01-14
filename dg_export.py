#!/usr/bin/env python3
"""DG Export helper script for hardware-optimized model export. #DG"""

import argparse

from ultralytics import YOLO


def parser_arguments():
    parser = argparse.ArgumentParser(description="Export YOLO model with hardware optimizations")
    parser.add_argument("--weights", type=str, required=True, help="initial weights path")
    parser.add_argument("--format", type=str, default="onnx", help="export format")
    parser.add_argument("--quantize", action="store_true", help="int8 export")
    parser.add_argument("--data", type=str, default="", help="dataset.yaml path for calibration")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="image size (pixels)")
    parser.add_argument("--no-separate-outputs", action="store_true", help="do not separate outputs")
    parser.add_argument("--separate-pose", action="store_true", help="separate pose outputs")
    parser.add_argument("--no-hw-optimized", action="store_true", help="disable hardware optimization")

    return parser.parse_args()


if __name__ == "__main__":
    args = parser_arguments()
    kwargs = vars(args)
    print(f"Export args: {kwargs}")

    model = YOLO(args.weights)

    export_kwargs = {
        "format": args.format,
        "simplify": True,
        "imgsz": args.imgsz,
        "int8": args.quantize,
        "separate_outputs": not args.no_separate_outputs,
        "export_hw_optimized": not args.no_hw_optimized,
        "separate_pose": args.separate_pose,
    }

    if args.data:
        export_kwargs["data"] = args.data

    success = model.export(**export_kwargs)
    print(f"Export completed: {success}")
