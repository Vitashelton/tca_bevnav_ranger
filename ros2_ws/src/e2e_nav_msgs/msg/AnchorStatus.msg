# Status of the pose anchor backend (state-estimation-agnostic interface).
std_msgs/Header header
string backend_type            # none|wheel_odom|fast_lio2|fast_livo|fast_livo2|external_odom|mock_odom
bool is_valid
float32 anchor_quality         # [0,1]
float32 odom_frequency         # Hz
float32 pose_jump_score        # higher = more jumps
float32 velocity_consistency   # [0,1], higher = better
float32 covariance_score       # normalized covariance metric
bool degraded_mode
string message
