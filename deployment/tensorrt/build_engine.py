#!/usr/bin/env python3
"""Build a TensorRT engine from the exported policy ONNX.

Target: Jetson Orin Nano (TensorRT 8.x, JetPack 5/6). FP16 by default.

Usage (on the Jetson):
  python3 build_engine.py --onnx policy.onnx --engine policy_fp16.plan --fp16

This uses the Python TensorRT API. It is written defensively because the API
surface differs slightly across TRT 8.0 -> 8.6. If `tensorrt` is not installed
(e.g. on the PC), the script explains how to run it on the Jetson rather than
fabricating an engine.
"""
import argparse
import os
import sys


def build(onnx_path, engine_path, fp16=True, workspace_mb=2048,
          input_shapes=None):
    try:
        import tensorrt as trt
    except Exception:
        print("ERROR: TensorRT not available in this environment.\n"
              "Run this script on the Jetson Orin Nano where JetPack provides\n"
              "the `tensorrt` python module, or inside the NGC TensorRT image.")
        return False

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("ONNX parse error:", parser.get_error(i))
            return False

    config = builder.create_builder_config()
    # workspace API differs across versions; try both.
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     workspace_mb * 1024 * 1024)
    except AttributeError:  # TRT < 8.4
        config.max_workspace_size = workspace_mb * 1024 * 1024

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build_engine] FP16 enabled")

    # Fixed shapes: define an optimization profile matching the ONNX inputs.
    profile = builder.create_optimization_profile()
    shapes = input_shapes or {
        'bev': (1, 13, 100, 100), 'goal': (1, 4), 'vel': (1, 3)}
    for name, shp in shapes.items():
        profile.set_shape(name, shp, shp, shp)
    config.add_optimization_profile(profile)

    try:
        serialized = builder.build_serialized_network(network, config)
    except AttributeError:  # very old API
        engine = builder.build_engine(network, config)
        serialized = engine.serialize()
    if serialized is None:
        print("[build_engine] FAILED to build engine.")
        return False
    os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized)
    print(f"[build_engine] wrote {engine_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--engine', default='policy_fp16.plan')
    ap.add_argument('--fp16', action='store_true', default=True)
    ap.add_argument('--fp32', dest='fp16', action='store_false')
    ap.add_argument('--workspace', type=int, default=2048, help='MB')
    args = ap.parse_args()
    ok = build(args.onnx, args.engine, fp16=args.fp16, workspace_mb=args.workspace)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
