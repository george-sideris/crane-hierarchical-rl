import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rclpy.duration import Duration
from rosidl_runtime_py.utilities import get_message
from tf2_ros.buffer import Buffer
BAG="/mnt/ssd/zed0_recal_far_20260720_203305"; T_WALL=1784579738.9138262
CLOUD="/zed_0/zed_node/point_cloud/cloud_registered"
BMIN=np.array([-5.364,-1.684]); BMAX=np.array([-3.364,5.316])
def rdr(topics=None):
    r=rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=BAG,storage_id="sqlite3"),rosbag2_py.ConverterOptions("",""))
    if topics: r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return r
TYPES={t.name:t.type for t in rdr().get_all_topics_and_types()}
TFMsg=get_message(TYPES["/tf"]); Cloud=get_message(TYPES[CLOUD])
buf=Buffer(cache_time=Duration(seconds=1e6)); best=None
r=rdr(["/tf","/tf_static",CLOUD])
while r.has_next():
    topic,data,t=r.read_next()
    if topic=="/tf_static":
        for tr in deserialize_message(data,TFMsg).transforms: buf.set_transform_static(tr,"bag")
    elif topic=="/tf":
        for tr in deserialize_message(data,TFMsg).transforms: buf.set_transform(tr,"bag")
    else:
        dt=abs(t/1e9-T_WALL)
        if best is None or dt<best[0]: best=(dt,data)
msg=deserialize_message(best[1],Cloud)
def mat(tf):
    q=tf.transform.rotation; x,y,z,w=q.x,q.y,q.z,q.w; tr=tf.transform.translation
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    M=np.eye(4); M[:3,:3]=R; M[:3,3]=[tr.x,tr.y,tr.z]; return M
stamp=Time.from_msg(msg.header.stamp)
base_mast=mat(buf.lookup_transform("base_link","mast",stamp))          # calib-independent
a=np.frombuffer(bytes(msg.data),np.float32).reshape(-1,msg.point_step//4)[:,:3]
a=a[np.isfinite(a).all(1)]; a=a[np.random.default_rng(0).choice(len(a),min(40000,len(a)),replace=False)]
ah=np.c_[a,np.ones(len(a))]
def floor_z(calib_path, tag):
    Tmo=np.load(calib_path)                       # T_mast<-optical
    P=(base_mast@Tmo@ah.T).T[:,:3]
    inxy=P[(P[:,0]>=BMIN[0])&(P[:,0]<=BMAX[0])&(P[:,1]>=BMIN[1])&(P[:,1]<=BMAX[1])]
    if len(inxy)<50: print(f"{tag:22s}: only {len(inxy)} in-box pts (calib puts cloud outside the box - stale/invalid here)"); return
    h,e=np.histogram(inxy[:,2],bins=np.arange(-1.7,0.2,0.03)); fz=e[np.argmax(h)]
    print(f"{tag:22s}: floor mode {fz:+.2f}  (in-box pts {len(inxy)}, z-min {inxy[:,2].min():+.2f})")
floor_z("/workspace/crane_testbed/calibration/out/calib_zed_0_mount.prebump_20260720.npy","PRE-bump (Jul-2)")
floor_z("/workspace/crane_testbed/calibration/out/calib_zed_0_mount.recal1_20260720.npy","first recal (near)")
floor_z("/workspace/crane_testbed/calibration/out/calib_zed_0_mount.npy","current (combined)")
print("crop/action floor is -1.30; sim floor ~-1.30")
