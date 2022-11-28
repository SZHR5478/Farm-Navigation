import numpy as np
import time
import rosproxy
from collections import deque
import matplotlib.pyplot as plt
import argparse

class UE4():
    def __init__(self, env_name, agent_name, master_ip, send_port, receive_port):
        self.HistoryImages = deque(maxlen=10)

        send_address = 'tcp://{}:{}'.format(master_ip, send_port)
        receive_address = 'tcp://{}:{}'.format(master_ip, receive_port)


        rosproxy.proxy.Setup(env_name, send_address, receive_address)

        #订阅rgba图像数据
        self.subcribe_RGBA = rosproxy.proxy.Topic()
        self.subcribe_RGBA.Init(agent_name, '/{}/{}/color/image_raw'.format(env_name, agent_name), 'sensor_msgs/Image', 10)
        self.subcribe_RGBA.Subscribe(self.callback_RGBA)

        #订阅点云数据
        self.subcribe_PointCloud = rosproxy.proxy.Topic()
        self.subcribe_PointCloud.Init(agent_name, '/{}/{}/points'.format(env_name, agent_name), 'sensor_msgs/PointCloud2', 10)
        self.subcribe_PointCloud.Subscribe(self.callback_PointCloud)

        #订阅里程计
        self.subcribe_Odom = rosproxy.proxy.Topic()
        self.subcribe_Odom.Init(agent_name, '/{}/{}/odom'.format(env_name, agent_name), 'nav_msgs/Odometry', 10)
        self.subcribe_Odom.Subscribe(self.callback_Odom)

        #订阅状态信息
        self.subcribe_State = rosproxy.proxy.Topic()
        self.subcribe_State.Init(agent_name, '/{}/{}/state'.format(env_name, agent_name), 'nav_msgs/Odometry', 10)
        self.subcribe_State.Subscribe(self.callback_State)

        #发布行动指令
        self.publish_vel = rosproxy.proxy.Topic()
        self.publish_vel.Init(agent_name, '/{}/{}/vel'.format(env_name, agent_name), 'geometry_msgs/Twist', 10)
        self.publish_vel.Advertise()

        #发布重置指令
        self.publish_reset = rosproxy.proxy.Topic()
        self.publish_reset.Init(agent_name, '/{}/{}/reset'.format(env_name, agent_name), 'geometry_msgs/Twist', 10)
        self.publish_reset.Advertise()

    # (540, 960, 4)
    def callback_RGBA(self, mes):
        self.CurrentImage = np.resize(np.array(list(mes.data)), (mes.height, mes.width, mes.step//mes.width))
        self.HistoryImages.append(self.CurrentImage)

    #获取小车位置?
    def callback_Odom(self, mes):
        self.x = mes.pose.pose.position.x
        self.y = mes.pose.pose.position.y
        self.z = mes.pose.pose.position.z

    def callback_PointCloud(self, mes):
        mes.header
        mes.height
        mes.width
        mes.fields
        mes.is_bigendian
        mes.point_step
        mes.row_step
        mes.is_dense

    def callback_State(self, mes):
        pass



    def plot_image(self):
        time.sleep(1)
        plt.imshow(self.CurrentImage)

    def set_reset(self):
        Twist = rosproxy.message.geometry_msgs.Twist()
        Twist.linear.z = 1
        self.publish_reset.Publish(Twist)


    #time.sleep(0.03)

    def set_move(self, angle, velocity):
        Twist = rosproxy.message.geometry_msgs.Twist()
        Twist.linear.x = float(velocity)
        Twist.angular.z = float(angle)
        self.publish_vel.Publish(Twist)

    def get_observation(self):
        state = self.HistoryImages
        return state

    def get_pos(self):
        time.sleep(1)
        return self.x, self.y,self.z

parser = argparse.ArgumentParser(description='A3C')

parser.add_argument('--env_name', default='Qiqihar')
parser.add_argument('--agent_name', default='T1204_00')
parser.add_argument('--master_ip', default='192.168.61.212')
parser.add_argument('--send_port', default=6658)
parser.add_argument('--receive_port', default=6587)

'''
if __name__ == '__main__':
    args = parser.parse_args()
    ue4 = UE4(args.env_name,args.agent_name,args.master_ip,args.send_port,args.receive_port)
    ue4.plot_image()
    print(ue4.get_pos())
'''