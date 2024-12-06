import os
import sys

sys.path.append( os.getcwd() )

# debugging 
import pdb 

# ---------------------------------- 
# ---------------------------------- 

import pinocchio as pin
from utils.meshcat_viewer_wrapper import MeshcatVisualizer
import time
import numpy as np
from numpy.linalg import inv, norm, pinv, svd, eig
from scipy.optimize import fmin_bfgs, fmin_slsqp
from utils.load_ur5_with_obstacles import load_ur5_with_obstacles, Target
import matplotlib.pylab as plt

# matplotlib with interactive setting
plt.ion()  

robot = load_ur5_with_obstacles(reduced=True)

viz = MeshcatVisualizer(robot)
viz.display(robot.q0)

target = Target(viz,position = np.array([.5,.5]))

# ---------------------------------- 
# ---------------------------------- 

def get_position_end_eff(q):
     '''
     Return the 2d position of the end effector.
     '''
     pin.framesForwardKinematics(robot.model,robot.data,q)
     return robot.data.oMf[-1].translation[[0,2]]

def compute_distance_end_target(q):
     '''
     Return the distance between the end effector end the target (2d).
     '''
     return norm(get_position_end_eff(q)-target.position)

def check_collision(q):
     '''
     Return true if in collision, false otherwise.
     '''
     pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)
     return pin.computeCollisions(robot.collision_model,robot.collision_data,False)

def sample_random_config(check=False):
    '''
    Return a random configuration. If check is True, this
    configuration is not is collision
    '''
    while True:
        q = np.random.rand(2)*6.4-3.2  # sample between -3.2 and +3.2.
        if not check or not check_collision(q):
            return q

# ---------------------------------- 

def find_min_collision_distance(q):
     '''
     Return the minimal distance between robot and environment. 
     '''
     
     pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)
     
     if pin.computeCollisions(robot.collision_model,robot.collision_data,False): 0
     
     idx = pin.computeDistances(robot.collision_model,robot.collision_data)
     
     return robot.collision_data.distanceResults[idx].min_distance

# ---------------------------------- 
# ---------------------------------- 

def sample_small_distance(threshold=5e-2, display=False):
     ''' 
     Sample a random free configuration where the distance is small enough. 
     '''
     
     while True:
          
          q = sample_random_config()
          
          if display:
               viz.display(q)
               time.sleep(1e-3)
               
          if not check_collision(q) and compute_distance_end_target(q)<threshold:
               return q
          
viz.display(sample_small_distance())

# ---------------------------------- 
# ---------------------------------- 

# Random descent: crawling from one free configuration to the target with random steps.
def randomDescent(q0 = None):
     
     q = sample_random_config(check=True) if q0 is None else q0
     hist = [ q.copy() ]
     
     for i in range(100):
          
          # Choose a random step and apply 
          dq = sample_random_config()*.1 
          qtry = q+dq 
                    
          # if distance decreases without collision ... 
          if compute_distance_end_target(q) > compute_distance_end_target(q+dq) and not check_collision(q+dq): 
               q = q+dq                 # keep the step
               hist.append(q.copy())    # keep a trace of it
               viz.display(q)           # display it
               time.sleep(5e-3)         # and sleep for a short while
               
     return hist 

# ---------------------------------- 
# ---------------------------------- 

def sampleSpace(nbSamples=500):
     '''
     Sample nbSamples configurations and store them in two lists depending
     if the configuration is in free space (hfree) or in collision (hcol), along
     with the distance to the target and the distance to the obstacles.
     '''
     hcol = []
     hfree = []
     for i in range(nbSamples):
          q = sample_random_config(False)
          if not check_collision(q):
               hfree.append( list(q.flat) + [ compute_distance_end_target(q), find_min_collision_distance(q) ])
          else:
               hcol.append(  list(q.flat) + [ compute_distance_end_target(q), 1e-2 ])
     return hcol,hfree

def plotConfigurationSpace(hcol,hfree,markerSize=20):
     '''
     Plot 2 "scatter" plots: the first one plot the distance to the target for 
     each configuration, the second plots the distance to the obstacles (axis q1,q2, 
     distance in the color space).
     '''
     htotal = hcol + hfree
     h=np.array(htotal)
     plt.subplot(2,1,1)
     plt.scatter(h[:,0],h[:,1],c=h[:,2],s=markerSize,lw=0)
     plt.title("Distance to the target")
     plt.colorbar()
     plt.subplot(2,1,2)
     plt.scatter(h[:,0],h[:,1],c=h[:,3],s=markerSize,lw=0)
     plt.title("Distance to the obstacles")
     plt.colorbar()

hcol,hfree = sampleSpace(100)
plotConfigurationSpace(hcol,hfree)

# ---------------------------------- 
# ---------------------------------- 

### Plot random trajectories in the same plot
qinit = np.array([-1.1, -3. ])
for i in range(100):
     traj = randomDescent(qinit)
     if compute_distance_end_target(traj[-1])<5e-2:
          print('We found a good traj!')
          break
traj = np.array(traj)

# Chose trajectory end to be in [-pi,pi]
qend = (traj[-1]+np.pi) % (2*np.pi) - np.pi

# Take the entire trajectory it modulo 2 pi
traj += (qend-traj[-1])

plt.plot(traj[:,0],traj[:,1],'r',lw=5) 

# ---------------------------------- 
# ---------------------------------- 

# %jupyter_snippet optim
def cost(q):
    """
    Cost function: distance to the target.
    """
    return compute_distance_end_target(q) ** 2


def constraint(q):
    """
    Constraint function: distance to the obstacle should be strictly positive.
    """
    min_collision_dist = 0.01  # [m]
    return find_min_collision_distance(q) - min_collision_dist


def callback(q):
    """
    At each optimization step, display the robot configuration.
    """
    viz.display(q)
    time.sleep(0.01)


def optimize():
     '''
     Optimize from an initial random configuration to discover a collision-free
     configuration as close as possible to the target. 
     '''
     return fmin_slsqp(x0=sample_random_config(check=True),
                       func=cost,
                       f_ieqcons=constraint,callback=callback,
                       full_output=1)
optimize()

while True:
    res=optimize()
    q=res[0]
    viz.display(q)
    if res[4]=='Optimization terminated successfully' and res[1]<1e-6:
        print('Finally successful!')
        break
    print("Failed ... let's try again! ")
    
# ---------------------------------- 
# KEEP SCRIPT RUNNING 
# ---------------------------------- 

# print("Keep Meshcat server alive")

# while True:
#     time.sleep(1)

print("show plot")
plt.show(block=True) 
