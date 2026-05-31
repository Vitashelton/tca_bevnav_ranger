#!/usr/bin/env bash
# Install runtime dependencies on a Jetson Orin Nano (JetPack 5/6, Ubuntu 22.04).
# TensorRT, CUDA and cuDNN are provided by JetPack -- do NOT pip install them.
set -e

echo "[install] checking JetPack-provided TensorRT..."
python3 -c "import tensorrt; print('TensorRT', tensorrt.__version__)" \
  || echo "WARNING: tensorrt not found. Install JetPack components first."

echo "[install] python deps for the runtime (no torch needed on edge)..."
python3 -m pip install --user numpy pyyaml onnxruntime || true

echo "[install] pycuda (needed by infer_trt.py)..."
python3 -m pip install --user pycuda || \
  echo "WARNING: pycuda build failed; ensure CUDA toolkit is on PATH."

echo "[install] ROS2 Humble is assumed already installed (source /opt/ros/humble/setup.bash)."
echo "[install] TODO: install Livox ROS2 driver (livox_ros_driver2) and"
echo "          realsense2_camera per vendor instructions; this script does not"
echo "          fabricate those installs."
echo "[install] done."
