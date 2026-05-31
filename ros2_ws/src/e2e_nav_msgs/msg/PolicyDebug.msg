# Raw policy output and runtime diagnostics.
std_msgs/Header header
float32 vx_raw
float32 vy_raw
float32 wz_raw
float32 uncertainty
float32 inference_time_ms
string runtime_type            # dummy|torch|onnx|tensorrt
bool policy_valid
