#!/usr/bin/python
import matplotlib
import matplotlib.animation as animation
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.use("TkAgg")
import rospy
from gvgexploration.msg import DataSize, Coverage


plt.rcParams['toolbar'] = 'None'
plt.rcParams["figure.frameon"] = False
methods = {'gvgexploration': 'GVGExp', 'continuous_connectivity': 'Continuous Connectivity'}
style.use('fivethirtyeight')

fig, axs = plt.subplots(2, 1, figsize=(10,8))


conn_t = [0]
conn = [0]
exp_t = [0]
exp = [0]
current_method = 'gvgexploration'
first_time = True


def plot_connection(data):
    conn_t.append(rospy.Time.now().to_sec())
    conn.append(conn[-1] + 1)


def plot_explored_area(data):
    exp_t.append(data.header.stamp.to_sec())
    exp.append(data.coverage)


def animate(i):
    axs[0].clear()
    axs[1].clear()
    if len(conn) == 1 or len(exp) == 1:
        axs[0].set_xlim(min(conn_t), max(conn_t) + 1)
        axs[0].set_ylim(min(conn), max(conn) + 1)
        axs[1].set_xlim(min(exp_t), max(exp_t) + 1)
        axs[1].set_ylim(min(exp), max(exp) + 1)
        first_time = False
    axs[0].step(conn_t, conn, color='green')
    axs[1].plot(exp_t, exp, 'y-o')
    axs[1].set_xlabel("Time(s)")
    axs[0].set_ylabel("Established Connections")
    axs[1].set_ylabel("Explored Ratio")
    axs[0].set_title(methods[current_method])


if __name__ == '__main__':
    rospy.init_node('comm_node')
    current_method = rospy.get_param("/method")
    rospy.Subscriber("/shared_data_size", DataSize, plot_connection)
    rospy.Subscriber("/coverage", Coverage, plot_explored_area)
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()
    rospy.spin()
