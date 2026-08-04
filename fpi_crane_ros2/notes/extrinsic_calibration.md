# offline camera-to-robot extrinsic calibration

We are looking for the parent link to camera optical frame transform. Spefically for the two stereo cameras in the setup:

    mast_link  -> zed0_left_camera_optical_frame
    stick_link -> zed1_left_camera_optical_frame

These transforms will then be hard-coded into the URDF.

## Problem Formulation

The problem can be expressed in the following form:

    unknown:  parent_link_T_camera_optical

    known from each bag sample:
    - image marker corner pixels
    - camera intrinsics
    - joint states
    - URDF/FK
    - measured 3D marker corner coordinates relative to gripper links

Given all the images of the markers, optimize the transform, $^{parent_link}T_{camera_optical}$ to minimize the total reprojection error over all poses, markers, and corners. We find the transform that is the "best fit" for all the data aquired. 

