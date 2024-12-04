# ----------------------------------
# SET UP 
# ----------------------------------

# set up the environment for the test 
import  pinocchio as pin 
from    utils.meshcat_viewer_wrapper import MeshcatVisualizer 
import  time 
import  numpy as np 
from    numpy.linalg import inv, norm, pinv, svd, eig 
from    scipy.optimize import fmin_bfgs, fmin_slsqp 
from    utils.load_ur5_with_obstacles import load_ur5_with_obstacles, Target 
import  matplotlib.pyplot as plt 

# load robot and display it (includes robot model and obstacles) 
robot = load_ur5_with_obstacles(reduced = True)
print("Robot loaded:", robot)

# # Print the fields of the robot variable
# print("Fields in robot:")
# for field in dir(robot):
#     print(field)

# initialize 3D viewer 
vis = MeshcatVisualizer(robot)
print("Visualizer initialized")

# Display the robot
vis.display(robot.q0)
print("Robot displayed") 

# display new configuration 
new_config = [3., -1.5]
vis.display( np.array( new_config ) )
print("New config = ", new_config) 

# set up target (as green dot) 
target_pos = np.array( [.5, .5] )
target = Target(vis, position = target_pos) 

# ----------------------------------  
# USING THE ROBOT MODEL 
# ---------------------------------- 

''' 
The robot is originally a 6 DOF manipulator 
    - We will only use joints 1 and 2 
    - Model loaded with "frozen" extra joints, which we will ignore 
    - Reload model with reduces = False to recover full model 
'''

# this function computes position of the end effector (in 2D) 
def find_position_end_effector(q): 
    ''' 
    Return the position of the end effector in 2D. 
    '''
    
    pin.framesForwardKinematics(robot.model, robot.data, q)
     
    position = robot.data.oMf[-1].translation[ [0, 2] ] 
    
    return position 

# this function checks if the robot is in collision, and returns True if collision is detected 
def check_collision(q): 
    '''
    Return True if robot is in collision, False otherwise.
    '''
    
    pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q) 
    
    is_collision = pin.computeCollisions(robot.model, robot.data, robot.collision_model, robot.collision_data, False) 
    
    return is_collision 

# this function computes the distance between the end effector and the target 
def compute_distance_target(q): 
    ''' 
    Return the distance between the end effector and the target (2d).
    '''
    
    return 0. 

# ----------------------------------
# RANDOM SEARCH OF VALID CONFIGURATION 
# ----------------------------------

# sample the configuration space until a free configuration is found 
def find_random_config(check = False): 
    ''' 
    Return a random configuration. If 'check' is True, then this configuration is valid, i.e. NOT in collision. 
    '''
    
    pass 

# ----------------------------------
# FROM RANDOM CONFIGURATION TO TARGET 
# ----------------------------------

# random descent: crawling from one free configuration to the target with random steps 
def compute_random_walk(q0 = None): 
    ''' 
    Make a random walk of 0.1 steps toward the target. 
    Return the list of configurations visited. 
    '''
    
    if q0 is None: 
        q = find_random_config(check = True) 
    else: 
        q = q0 
        
    hist = [ q.copy() ] 
    
    return hist 

compute_random_walk() 
    
# ----------------------------------
# KEEP SCRIPT RUNNING 
# ----------------------------------

print("Keep Meshcat server alive")

# Keep the script running to keep the Meshcat server alive
while True:
    time.sleep(1)
    

    