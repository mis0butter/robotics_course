# Stand-alone program to optimize the configuration q=[q1,q2] of a 2-R robot with scipy BFGS.

# ---------------------------------- 

import os 
import sys 

sys.path.append( os.getcwd() ) 

import pdb 

# ---------------------------------- 

import time
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from utils.meshcat_viewer_wrapper import MeshcatVisualizer, translation2d,planar
from numpy.linalg import norm, inv, pinv, svd, eig

# ---------------------------------- 
# Displaying objects and create a 2 DOF robot 
# ---------------------------------- 

viz = MeshcatVisualizer(url='classical')

viz.addSphere('joint1',.1,[1,0,0,1])
viz.addSphere('joint2',.1,[1,0,0,1])
viz.addSphere('joint3',.1,[1,0,0,1])
viz.addCylinder('arm1',.75,.05,[.65,.65,.65,1])
viz.addCylinder('arm2',.75,.05,[.65,.65,.65,1])
viz.addSphere('target',.1001,[0,.8,.1,1])

def display(q):
    '''Display the robot in Gepetto Viewer. '''
    assert (q.shape == (2,))
    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[0] + q[1])
    s1 = np.sin(q[0] + q[1])
    viz.applyConfiguration('joint1',planar(0,           0,           0))
    viz.applyConfiguration('arm1'  ,planar(c0 / 2,      s0 / 2,      q[0]))
    viz.applyConfiguration('joint2',planar(c0,          s0,          q[0]))
    viz.applyConfiguration('arm2'  ,planar(c0 + c1 / 2, s0 + s1 / 2, q[0] + q[1]))
    viz.applyConfiguration('joint3',planar(c0 + c1,     s0 + s1,     q[0] + q[1]))

# ---------------------------------- 

def get_endeffector(q):    
    '''
    Return the 2D position of the end effector of the robot at configuration q. 
    '''
    
    assert (q.shape == (2,))
    c0 = np.cos(q[0])
    s0 = np.sin(q[0])
    c1 = np.cos(q[0] + q[1])
    s1 = np.sin(q[0] + q[1])
    
    return np.array([c0 + c1, s0 + s1])

target = np.array([.5, .5])
viz.applyConfiguration('target',translation2d(target[0],target[1]))
                            
# ---------------------------------- 
# Optimize the configuration 
# ---------------------------------- 

def compute_cost(q):
    eff = get_endeffector(q)
    return np.linalg.norm(eff - target)**2

def display_callback(q):
    display(q)
    time.sleep(.5)

q0 = np.array([0.0, 0.0])
qopt_bfgs = fmin_bfgs(compute_cost, q0, callback=display_callback)
print('\n *** Optimal configuration from BFGS = %s \n\n\n\n' % qopt_bfgs)
