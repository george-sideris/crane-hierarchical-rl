import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rclpy.duration import Duration
from rosidl_runtime_py.utilities import get_message
from tf2_ros.buffer import Buffer
BAG="/mnt/ssd/zed0_recal_far_20260720_203305"; T_WALL=1784579738.9138262
BMIN=np.array([-5.364,-1.684,-1.30]); BMAX=np.array([-3.364,5.316,0.10])
CLOUD="/zed_0/zed_node/point_cloud/cloud_registered"
def rdr(topics=None):
    r=rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=BAG,storage_id="sqlite3"),rosbag2_py.ConverterOptions("",""))
    if topics: r.set_filter(rosbag2_py.StorageFilter(topics=topics))
    return r
TYPES={t.name:t.type for t in rdr().get_all_topics_and_types()}
TFMsg=get_message(TYPES["/tf"]); Cloud=get_message(TYPES[CLOUD])
buf=Buffer(cache_time=Duration(seconds=1e6))
best=None; r=rdr(["/tf","/tf_static",CLOUD])
while r.has_next():
    topic,data,t=r.read_next()
    if topic=="/tf_static":
        for tr in deserialize_message(data,TFMsg).transforms: buf.set_transform_static(tr,"bag")
    elif topic=="/tf":
        for tr in deserialize_message(data,TFMsg).transforms: buf.set_transform(tr,"bag")
    else:
        dt=abs(t/1e9-T_WALL)
        if best is None or dt<best[0]: best=(dt,data,t)
msg=deserialize_message(best[1],Cloud)
def quatmat(q):
    x,y,z,w=q.x,q.y,q.z,q.w
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
tf=buf.lookup_transform("base_link",msg.header.frame_id,Time.from_msg(msg.header.stamp))
R=quatmat(tf.transform.rotation); tr=tf.transform.translation
a=np.frombuffer(bytes(msg.data),np.float32).reshape(-1,msg.point_step//4)[:,:3]
a=a[np.isfinite(a).all(1)]
P=a@R.T+np.array([tr.x,tr.y,tr.z])
print(f"raw cloud in base: {len(P)} pts")
# within the XY footprint of the box, what's the z-profile? is there log below the crop floor?
inxy=P[(P[:,0]>=BMIN[0])&(P[:,0]<=BMAX[0])&(P[:,1]>=BMIN[1])&(P[:,1]<=BMAX[1])]
print(f"in box XY footprint: {len(inxy)} pts, z[{inxy[:,2].min():.2f},{inxy[:,2].max():.2f}]")
cur=((inxy[:,2]>=BMIN[2])&(inxy[:,2]<=BMAX[2])).sum()
print(f"  kept by current z-crop [-1.30,0.10]: {cur}")
for zlo in [-1.35,-1.40,-1.45,-1.50]:
    print(f"  if z-min lowered to {zlo}: {((inxy[:,2]>=zlo)&(inxy[:,2]<=BMAX[2])).sum()} pts  (+{((inxy[:,2]>=zlo)&(inxy[:,2]<BMIN[2])).sum()} below current floor)")
# front edge: points just beyond x-max (toward crane)
for xhi in [-3.36,-3.20,-3.00,-2.80]:
    m=(P[:,0]<=xhi)&(P[:,0]>=BMIN[0])&(P[:,1]>=BMIN[1])&(P[:,1]<=BMAX[1])&(P[:,2]>=BMIN[2])&(P[:,2]<=BMAX[2])
    print(f"  if x-max extended to {xhi}: {m.sum()} pts in-box")

# WHERE is the floor plane? histogram z of in-box points
print("\nz-histogram of in-box XY points (find the floor level):")
h,edges=np.histogram(inxy[:,2], bins=np.arange(-1.6,0.2,0.05))
for c,e in zip(h,edges):
    if c>20: print(f"  z {e:+.2f}..{e+0.05:+.2f}: {'#'*(c//30)} {c}")
floor_z=edges[np.argmax(h)]
print(f"dominant (floor) level ~{floor_z:.2f}  (crop floor is -1.30; sim/training floor ~-1.30)")
print(f"structure ABOVE the floor (logs): {((inxy[:,2]>floor_z+0.05)&(inxy[:,2]<0)).sum()} pts from {floor_z+0.05:.2f} up")
