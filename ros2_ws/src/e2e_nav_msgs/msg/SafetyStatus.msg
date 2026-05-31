# Safety supervisor decision and limited command.
std_msgs/Header header
bool safe
bool estop
bool obstacle_too_close
bool input_timeout
float32 min_obstacle_distance
float32 vx_limited
float32 vy_limited
float32 wz_limited
string reason
