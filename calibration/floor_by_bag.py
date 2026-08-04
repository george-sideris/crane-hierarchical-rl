import numpy as np, sys
import rosbag2_py
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rclpy.duration import Duration
from rosidl_runtime_py.utilities import get_message
from tf2_ros.buffer import Buffer
BMIN=np.array([-5.364,-1.684]); BMAX=np.array([-3.364,5.316])
CLOUD="/zed_0/zed_node/point_cloud/cloud_registered"
def floor(BAG):
    def rdr(topics=None):
        r=rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=BAG,storage_id="sqlite3"),rosbag2_py.ConverterOptions("",""))
        if topics: r.set_filter(rosbag2_py.StorageFilter(topics=topics))
        return r
    TYPES={t.name:t.type for t in rdr().get_all_topics_and_types()}
    TFMsg=get_message(TYPES["/tf"]); Cloud=get_message(TYPES[CLOUD])
    buf=Buffer(cache_time=Duration(seconds=1e6)); clouds=[]
    r=rdr(["/tf","/tf_static",CLOUD])
    while r.has_next():
        topic,data,t=r.read_next()
        if topic=="/tf_static":
            for tr in deserialize_message(data,TFMsg).transforms: buf.set_transform_static(tr,"bag")
        elif topic=="/tf":
            for tr in deserialize_message(data,TFMsg).transforms: buf.set_transform(tr,"bag")
        else: clouds.append((t,data))
    def mat(tf):
        q=tf.transform.rotation; x,y,z,w=q.x,q.y,q.z,q.w; tr=tf.transform.translation
        R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]); M=np.eye(4); M[:3,:3]=R; M[:3,3]=[tr.x,tr.y,tr.z]; return M
    fzs=[]
    for t,data in clouds[len(clouds)//3::max(1,len(clouds)//6)][:5]:
        msg=deserialize_message(data,Cloud)
        try: T=mat(buf.lookup_transform("base_link",msg.header.frame_id,Time.from_msg(msg.header.stamp)))
        except Exception: continue
        a=np.frombuffer(bytes(msg.data),np.float32).reshape(-1,msg.point_step//4)[:,:3]; a=a[np.isfinite(a).all(1)]
        P=(np.c_[a,np.ones(len(a))]@T.T)[:,:3]
        inxy=P[(P[:,0]>=BMIN[0])&(P[:,0]<=BMAX[0])&(P[:,1]>=BMIN[1])&(P[:,1]<=BMAX[1])]
        if len(inxy)>200:
            h,e=np.histogram(inxy[:,2],bins=np.arange(-1.7,0.3,0.03)); fzs.append(e[np.argmax(h)])
    print(f"{BAG.split('/')[-1]}: floor mode {np.median(fzs):+.2f} (n={len(fzs)} clouds)" if fzs else f"{BAG.split('/')[-1]}: no floor")
for b in sys.argv[1:]: floor(b)
