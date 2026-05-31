#!/usr/bin/env python3
"""Minimal TensorRT inference wrapper for the TCA-BEV policy.

Exposes a TrtPolicy class with .infer(bev, goal, vel) -> [vx, vy, wz, unc].
Used by e2e_policy_runtime when runtime_type == 'tensorrt' on the Jetson.

Requires `tensorrt` and `pycuda`. On non-Jetson machines these are usually
absent; the class raises a clear error so callers can fall back to ONNX/torch.
"""
import numpy as np


class TrtPolicy:
    def __init__(self, engine_path,
                 input_names=('bev', 'goal', 'vel'), output_name='action'):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401  (initializes CUDA context)
        self.cuda = cuda
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_names = input_names
        self.output_name = output_name
        self.stream = cuda.Stream()
        self._alloc()

    def _alloc(self):
        self.bindings = [None] * self.engine.num_bindings
        self.host = {}
        self.dev = {}
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            size = int(np.prod(shape))
            host = self.cuda.pagelocked_empty(size, np.float32)
            dev = self.cuda.mem_alloc(host.nbytes)
            self.bindings[i] = int(dev)
            self.host[name] = (host, shape)
            self.dev[name] = dev

    def infer(self, bev, goal, vel):
        feeds = {'bev': bev, 'goal': goal, 'vel': vel}
        for name in self.input_names:
            host, shape = self.host[name]
            host[:] = np.ascontiguousarray(feeds[name], np.float32).ravel()
            self.cuda.memcpy_htod_async(self.dev[name], host, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        out_host, out_shape = self.host[self.output_name]
        self.cuda.memcpy_dtoh_async(out_host, self.dev[self.output_name], self.stream)
        self.stream.synchronize()
        return np.array(out_host).reshape(out_shape)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True)
    args = ap.parse_args()
    try:
        p = TrtPolicy(args.engine)
        out = p.infer(np.random.randn(1, 13, 100, 100).astype(np.float32),
                      np.random.randn(1, 4).astype(np.float32),
                      np.random.randn(1, 3).astype(np.float32))
        print("output:", out)
    except Exception as e:
        print("TensorRT inference unavailable here:", e)
