string reference_name

bool valid
string status

# Reference-up vector expressed in base_link
geometry_msgs/Vector3 reference_z_base

float64 dot_product
float64 tilt_rad
float64 tilt_deg
float64 reward

# Nonzero only for an IMU-derived result
uint32 samples_used