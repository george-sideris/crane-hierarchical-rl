#gripper_move_to	0 = do nothing on the gripper    1 =open the gripper  2 = close the gripper
#isExecuted            is executed by the sequencer manager 
#delta_time  		time of motion max for the action in MilliSec (exit condition)
#delta_angle 		angle of motion (exit condition)

int32 sequence_id
bool  is_executed
int32 gripper_move_to
int32 delta_time
int32 delta_angle
