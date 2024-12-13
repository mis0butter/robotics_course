# Stand-alone program to optimize the placement of a 2d robot, 
# where the decision variables are the placement of the 3 bodies 
# of the robot. BFGS and SLSQP solvers are used.

# ---------------------------------- 

import os 
import sys 

sys.path.append( os.getcwd() ) 

import pdb 

# ---------------------------------- 

import time 
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp 
from utils.meshcat_viewer_wrapper import MeshcatVisualizer,translation2d, planar 
from numpy.linalg import norm, inv, pinv, svd, eig 

# ---------------------------------- 
# DISPLAYING OBJECTS 
# ---------------------------------- 

viz = MeshcatVisualizer()
# viz = MeshcatVisualizer(url = 'classical')

viz.addSphere( 'joint1', .1, [1,0,0,1] ) 
viz.addSphere( 'joint2', .1, [1,0,0,1] ) 
viz.addSphere( 'joint3', .1, [1,0,0,1] ) 
viz.addCylinder( 'arm1', .75,.05, [.65,.65,.65,1] ) 
viz.addCylinder( 'arm2', .75,.05, [.65,.65,.65,1] ) 
viz.addSphere( 'target', .1001, [0,.8,.1,1] ) 

# ---------------------------------- 

def display_9(pose):
    ''' 
    Display the robot in the Viewer. 
    '''
    
    assert (pose.shape == (9, ))
    
    x1, y1, theta1, x2, y2, theta2, x3, y3, theta3 = pose 
    
    dx1 = np.cos(theta1) ; dy1 = np.sin(theta1) 
    dx2 = np.cos(theta2) ; dy2 = np.sin(theta2) 
    
    viz.applyConfiguration( 'joint1', planar(x1,           y1,           theta1) )
    viz.applyConfiguration( 'arm1'  , planar(x1 + dx1 / 2, y1 + dy1 / 2, theta1) )
    viz.applyConfiguration( 'joint2', planar(x2,           y2,           theta2) )
    viz.applyConfiguration( 'arm2'  , planar(x2 + dx2 / 2, y2 + dy2 / 2, theta2) )
    viz.applyConfiguration( 'joint3', planar(x3,           y3,           theta3) )

# test function 
display_9( np.array( [0,0,0,1,1,0,2,2,0] ) ) 

# ---------------------------------- 

def get_endeffector_9(pose):
    assert (pose.shape == (9, ))
    x1, y1, t1, x2, y2, t2, x3, y3, t3 = pose
    return np.array([x3, y3])


target = np.array([.5, .5])
viz.applyConfiguration('target',translation2d(target[0],target[1]))

# ---------------------------------- 

def compute_cost_9(pose):
    eff = get_endeffector_9(pose)
    return norm(eff - target)**2 

def constraint_9(pose):
    
    assert (pose.shape == (9, ))
    x1, y1, t1, x2, y2, t2, x3, y3, t3 = pose
    
    constraints = np.zeros(6)
    constraints[0] = x1 - 0
    constraints[1] = y1 - 0
    constraints[2] = x1 + np.cos(t1) - x2
    constraints[3] = y1 + np.sin(t1) - y2
    constraints[4] = x2 + np.cos(t2) - x3
    constraints[5] = y2 + np.sin(t2) - y3
    
    return constraints 

# ---------------------------------- 

def compute_penalty(pose):
    return compute_cost_9(pose) + 10 * sum(np.square(constraint_9(pose)))

def display_callback_9(pose):
    display_9(pose)
    time.sleep(.5)

x0 = np.array([ 0.0,] * 9)

with_bfgs = 0
if with_bfgs:
    xopt = fmin_bfgs(
        compute_penalty, 
        x0, 
        callback = display_callback_9
        )
else:
    xopt = fmin_slsqp(
        compute_cost_9, 
        x0, 
        callback = display_callback_9, 
        f_eqcons = constraint_9, 
        iprint = 2, 
        full_output = 1
        )[0]
print('\n *** Xopt = %s\n\n\n\n' % xopt)

# ---------------------------------- 
# KEEP SCRIPT RUNNING 
# ---------------------------------- 

print("Keep Meshcat server alive") 

while True: 
    time.sleep(1)
