#!/usr/bin/env python3
"""Compare ONNX Runtime vs TensorRT engine outputs on random inputs.

Usage (Jetson):
  python3 compare_onnx_trt.py --onnx policy.onnx --engine policy_fp16.plan

Reports max/mean absolute error. FP16 engines typically show errors ~1e-2 to
1e-3 relative to FP32 ONNX, which is acceptable for velocity commands; large
errors indicate a layer was not supported and silently fell back.
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--engine', required=True)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--channels', type=int, default=13)
    ap.add_argument('--size', type=int, default=100)
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except Exception:
        print("onnxruntime missing. pip install onnxruntime")
        return
    from infer_trt import TrtPolicy

    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    trt_policy = TrtPolicy(args.engine)

    maxe, meane = [], []
    for _ in range(args.trials):
        bev = np.random.randn(1, args.channels, args.size, args.size).astype(np.float32)
        goal = np.random.randn(1, 4).astype(np.float32)
        vel = np.random.randn(1, 3).astype(np.float32)
        ref = sess.run(['action'], {'bev': bev, 'goal': goal, 'vel': vel})[0]
        out = trt_policy.infer(bev, goal, vel)
        maxe.append(float(np.max(np.abs(ref - out))))
        meane.append(float(np.mean(np.abs(ref - out))))
    print(f"[compare_onnx_trt] max_err={max(maxe):.3e} mean_err={np.mean(meane):.3e}")


if __name__ == '__main__':
    main()
