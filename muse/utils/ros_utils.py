import importlib
import sys
import threading

from dotmap import DotMap


# this should be called before all ros imports
def ros_import_open():
    if importlib.find_loader('rospy') is None:
        sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages")


# this should be called after all ros imports are done
def ros_import_close():
    try:
        sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
    except Exception as e:
        pass

ros_import_open()
import rospy
ros_import_close()

class RosTopic:
    def __init__(self, topic, type):
        self._topic = topic
        self._type = type

    @property
    def topic(self):
        return self._topic

    @property
    def type(self):
        return self._type

    def __eq__(self, other):
        return other.topic == self.topic and other.type == self.type

    def __hash__(self):
        return hash("%s %s" % (self._topic, self._type))


############ SINGLE ROS INSTANCE ###########

_ros_initialized = False
_ros_topic_pub_map = dict()
_ros_topic_sub_list = []
_rate = None
_background_thread = None


#  called once
def setup_ros_node(name, **kwargs):
    global _ros_initialized
    assert not _ros_initialized, 'ROS was already intialized'
    rospy.init_node(name, **kwargs)
    _ros_initialized = True


#  returns tuple (was_created, publisher)
def get_publisher(rt: RosTopic, **pub_kwargs):
    created = False
    if rt not in _ros_topic_pub_map.keys():
        created = True
        _ros_topic_pub_map[rt] = rospy.Publisher(rt.topic, rt.type, **pub_kwargs)

    return created, _ros_topic_pub_map[rt]


#  returns bound subscriber, may need to pass in callback_args or queue_size
def bind_subscriber(rt: RosTopic, sub_cb, **sub_kwargs):
    sub = rospy.Subscriber(rt.topic, rt.type, sub_cb, **sub_kwargs)
    _ros_topic_sub_list.append(DotMap({"top": rt, "sub": sub}))
    return sub


# calls loop function(rate) periodically as determined by rate in separate thread
def setup_background_thread(loop_fn, rate):
    global _rate
    _rate = rate

    def _thread_fn():
        while not rospy.is_shutdown():
            # inner loop to publish stationary messages in between rollouts
            loop_fn()
            _rate.sleep()

    global _background_thread
    _background_thread = threading.Thread(target=_thread_fn)


# begins background thread
def start_background_thread():
    assert _background_thread is not None, "No background thread instantiated"
    _background_thread.start()
