#!/usr/bin/env python3
"""DG Benchmark helper script for model performance testing. #DG"""

import argparse

from ultralytics.utils.benchmarks import benchmark


def parser_arguments():
    parser = argparse.ArgumentParser(description="Benchmark YOLO model")
    parser.add_argument("--weights", type=str, required=True, help="model weights path")
    parser.add_argument("--data", type=str, default="coco.yaml", help="dataset.yaml path")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="image size (pixels)")
    parser.add_argument("--format", type=str, default="", help="specific format to benchmark (empty for all)")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision")
    parser.add_argument("--int8", action="store_true", help="use INT8 precision")
    parser.add_argument("--no-separate-outputs", action="store_true", help="disable separate outputs")
    parser.add_argument("--no-hw-optimized", action="store_true", help="disable hardware optimization")
    parser.add_argument("--verbose", action="store_true", help="verbose output")

    return parser.parse_args()


if __name__ == "__main__":
    args = parser_arguments()
    kwargs = vars(args)
    print(f"Benchmark args: {kwargs}")

    # Run benchmark
    results = benchmark(
        model=args.weights,
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        int8=args.int8,
        format=args.format,
        verbose=args.verbose,
        separate_outputs=not args.no_separate_outputs,
        export_hw_optimized=not args.no_hw_optimized,
    )

    print(f"\nBenchmark Results:\n{results}")
