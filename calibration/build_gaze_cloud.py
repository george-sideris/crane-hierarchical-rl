"""Build the REAL zed_0 gaze cloud in the crane BASE frame from the new static gaze bag.
P_base = T_base<-mast(slew) @ T_mast<-zed0_optical(calib) @ T_optical<-body(tf) @ P_body
Saves out/pcd_base_gaze.npy  (Nx3, base frame).
"""
import numpy as np, sys, struct
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import calib_fk as fk

TS = get_typestore(Stores.ROS2_FOXY)
BAG = sys.argv[1] if len(sys.argv) > 1 else "../rosbag2_cranelab_test_2026_06_22-20_31_25"
CAM = "zed_0"
CLOUD_FRAME = "zed_0_left_camera_frame"
OPT_CHILD = CLOUD_FRAME + "_optical"

def q2R(x,y,z,w):
    n=np.sqrt(x*x+y*y+z*z+w*w); x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def tf2T(tr):
    T=np.eye(4); q=tr.transform.rotation; t=tr.transform.translation
    T[:3,:3]=q2R(q.x,q.y,q.z,q.w); T[:3,3]=[t.x,t.y,t.z]; return T

opt_edge=None; slew=None; cloud_msg=None; densest=0
with Reader(BAG) as r:
    for con,ts,raw in r.messages():
        if con.topic in ('/tf','/tf_static'):
            m=TS.deserialize_cdr(raw,con.msgtype)
            for tr in m.transforms:
                if (tr.header.frame_id,tr.child_frame_id)==(CLOUD_FRAME,OPT_CHILD):
                    opt_edge=tf2T(tr)
        elif con.topic=='/joint_states' and slew is None:
            m=TS.deserialize_cdr(raw,con.msgtype)
            d=dict(zip(m.name,m.position)); slew=d['slew_joint']
            gaze={k:d[k+'_joint'] for k in ['slew','boom','stick','telescope','hanger','bearingfork','grapplecarrier']}
        elif con.topic==f'/{CAM}/zed_node/point_cloud/cloud_registered':
            m=TS.deserialize_cdr(raw,con.msgtype)
            # keep the densest (most finite) cloud
            arr=np.frombuffer(m.data,np.uint8).reshape(-1,m.point_step)
            xyz=arr[:,:12].copy().view(np.float32).reshape(-1,3)
            fin=np.isfinite(xyz).all(1).sum()
            if fin>densest: densest=fin; cloud_msg=(m.point_step,m.data,m.width,m.height)

step,data,w,h=cloud_msg
arr=np.frombuffer(data,np.uint8).reshape(-1,step)
P=arr[:,:12].copy().view(np.float32).reshape(-1,3).astype(np.float64)
P=P[np.isfinite(P).all(1)]
print(f"raw cloud {w}x{h}, finite pts {P.shape[0]}, frame {CLOUD_FRAME}")
print(f"gaze slew={slew:.4f}  joints={ {k:round(v,4) for k,v in gaze.items()} }")

calib=np.load(f"out/calib_{CAM}_mount.npy")          # T_mast<-zed_0_optical
T_base_mast=fk.joint_T('slew',slew)                  # base_link <- mast (slew only)
opt_inv=np.linalg.inv(opt_edge)                      # T_optical<-body
T=T_base_mast @ calib @ opt_inv
Pb=(T @ np.c_[P,np.ones(len(P))].T).T[:,:3]
np.save("out/pcd_base_gaze.npy",Pb.astype(np.float32))
print(f"saved out/pcd_base_gaze.npy  {Pb.shape[0]} pts")
print(f"  X[{Pb[:,0].min():.2f},{Pb[:,0].max():.2f}] Y[{Pb[:,1].min():.2f},{Pb[:,1].max():.2f}] Z[{Pb[:,2].min():.2f},{Pb[:,2].max():.2f}]")
print("opt_edge translation:",np.round(opt_edge[:3,3],4),"  calib trans:",np.round(calib[:3,3],4))
