# Time- and Calibration-Aware BEV tensor for the policy network.
# data is laid out as C,H,W (channel-major) and flattened row by row.
std_msgs/Header header
uint32 height
uint32 width
uint32 channels
float32 resolution        # meters per cell
float32 origin_x          # BEV origin (meters) in the reference frame
float32 origin_y
float32[] data            # length = channels*height*width, C,H,W order

# global / per-modality confidences in [0,1]
float32 global_confidence
float32 lidar_confidence
float32 depth_confidence
float32 calibration_confidence
float32 time_confidence
float32 anchor_quality
