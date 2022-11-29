import numpy as np
import rosproxy
from collections import deque

class UE4():
    def __init__(self, env_name, agent_name, master_ip, send_port, receive_port, historical_image_length):
        self.HistoryImages = deque(maxlen=historical_image_length)

        send_address = 'tcp://{}:{}'.format(master_ip, send_port)
        receive_address = 'tcp://{}:{}'.format(master_ip, receive_port)

        rosproxy.proxy.Setup(env_name, send_address, receive_address)

        #订阅rgba图像数据
        self.subcribe_RGBA = rosproxy.proxy.Topic()
        self.subcribe_RGBA.Init(agent_name, '/{}/{}/color/image_raw'.format(env_name, agent_name), 'sensor_msgs/Image', 10)
        self.subcribe_RGBA.Subscribe(self.callback_RGBA)

        #订阅里程计,获取智能体位置信息
        self.subcribe_Odom = rosproxy.proxy.Topic()
        self.subcribe_Odom.Init(agent_name, '/{}/{}/odom'.format(env_name, agent_name), 'nav_msgs/Odometry', 10)
        self.subcribe_Odom.Subscribe(self.callback_Odom)

        #发布行动指令
        self.publish_vel = rosproxy.proxy.Topic()
        self.publish_vel.Init(agent_name, '/{}/{}/vel'.format(env_name, agent_name), 'geometry_msgs/Twist', 10)
        self.publish_vel.Advertise()

        #发布重置指令
        self.publish_reset = rosproxy.proxy.Topic()
        self.publish_reset.Init(agent_name, '/{}/{}/reset'.format(env_name, agent_name), 'geometry_msgs/Twist', 10)
        self.publish_reset.Advertise()

    # 获取RGBA图像(540, 960, 4)
    def callback_RGBA(self, mes):
        self.CurrentImage = np.resize(np.array(list(mes.data)), (mes.height, mes.width, mes.step//mes.width))
        self.HistoryImages.append(self.CurrentImage)

    #获取小车位置，是通过里程计数据来获取位置信息吗?
    def callback_Odom(self, mes):
        self.x = mes.pose.pose.position.x
        self.y = mes.pose.pose.position.y
        self.z = mes.pose.pose.position.z

    def set_reset(self):
        Twist = rosproxy.message.geometry_msgs.Twist()
        Twist.linear.z = 1
        self.publish_reset.Publish(Twist)


    def set_move(self, angle, velocity):
        Twist = rosproxy.message.geometry_msgs.Twist()
        Twist.linear.x = float(velocity)
        Twist.angular.z = float(angle)
        self.publish_vel.Publish(Twist)

    def get_observation(self):
        state = np.array(self.HistoryImages)
        return state

    def get_pos(self):
        return np.array([self.x, self.y,self.z])