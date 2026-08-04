#plc_crane_move_id		       the crane move sequence id
#plc_crane_move_point_id              current point id in execution by the PLC
#plc_trajectory_count_in_queue        number of trajectory in the plc node queue

#plc_crane_state		       the crane move sequence state (0=INIT, 1=HOLD, 2=START_SEQUENCE, 3=EXECUTE_SEQUENCE, 4=END_SEQUENCE)

#plc_gripper_move_id                  the gripper move sequence id 
#plc_gripper_move_completeness        the gripper move execution in percent    


uint32 plc_crane_sequence_id
uint32 plc_crane_move_point_id
uint32 plc_trajectory_count_in_queue
uint32 plc_crane_state
uint32 plc_gripper_sequence_id
uint32 plc_gripper_move_completeness
uint32 plc_gripper_move_angle
uint32 plc_gripper_state


