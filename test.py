# ----------------------------------
# SET UP 
# ----------------------------------

# set up environment for test 
import  pinocchio as pin 
from    utils.meshcat_viewer_wrapper import MeshcatVisualizer 
import  time 
import  numpy as np 
from    numpy.linalg import inv, norm, pinv, svd, eig 
from    scipy.optimize import fmin_bfgs, fmin_slsqp 
from    utils.load_ur5_with_obstacles import load_ur5_with_obstacles, Target 
import  matplotlib.pyplot as plt 

# debugging 
import pdb 
import IPython 

# load robot and display - includes model and obstacles 
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

# Robot is originally a 6 DOF manipulator 
#     - We will only use joints 1 and 2 
#     - Model loaded with "frozen" extra joints (ignored)  
#     - Reload with reduced = False for full model  

# ---------------------------------- 

def get_position_end_eff(q): 
    ''' 
    Return the position of the end effector in 2D. 
    '''
    
    pin.framesForwardKinematics(robot.model, robot.data, q)
     
    position = robot.data.oMf[-1].translation[ [0, 2] ] 
    
    return position 

# test function 
position = get_position_end_eff(robot.q0) 
print("End effector position = ", position ) 
 
# ---------------------------------- 

def check_collision(q): 
    '''
    Return True if robot is in collision, False otherwise.
    '''
    
    pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q) 
    
    is_collision = pin.computeCollisions(robot.collision_model, robot.collision_data, False)
    
    return is_collision 

# test function 
collision = check_collision(robot.q0) 
# print("Collision = ", collision) 

# ---------------------------------- 

def compute_distance_target(q): 
    ''' 
    Return the distance between the end effector and the target (2d).
    '''
    
    pos_end_eff = get_position_end_eff(q) 
    target_pos  = target.position  
    distance    = norm( pos_end_eff - target_pos )  
    
    return distance 

# test function 
distance = compute_distance_target(robot.q0) 
print("Distance to target = ", distance) 

# ---------------------------------- 
# RANDOM SEARCH OF VALID CONFIGURATION 
# ---------------------------------- 

def sample_valid_config(check = False): 
    ''' 
    Sample the configuration space until a free configuration is found. 
        - If 'check' is True, then this configuration is valid, i.e. NOT in collision. 
        - If 'check' is False, then this configuration is not checked for collision. 
    ''' 
    
    while True: 
    
        # sample between -3.2 and +3.2 
        q_random = np.random.rand(2)*6.4-3.2  
        collision = check_collision(q_random) 
        
        # if "don't check" OR "no collision," use config 
        if not check or not collision: 
            return q_random
       
# test function 
q = sample_valid_config(check = True) 
print("Random configuration = ", q) 

vis.display(q) 

# ----------------------------------
# FROM RANDOM CONFIGURATION TO TARGET 
# ----------------------------------

def make_random_descent(q0 = None): 
    ''' 
    Make a random walk of 0.1 steps toward the target. 
    Return the list of configurations visited. 
    '''
    
    if q0 is None: 
        q = sample_valid_config(check = True) 
    else: 
        q = q0 
        
    hist = [ q.copy() ] 
        
    for i in range(100): 
        
        # choose a random step 
        dq      = sample_valid_config() * 0.1 
        qtry    = q + dq 
        
        distance_q     = compute_distance_target(q) 
        distance_qtry  = compute_distance_target(qtry) 
        collision_qtry = check_collision(qtry) 
        
        if distance_q > distance_qtry and not collision_qtry: 
            q = qtry 
            hist.append( q.copy() ) 
            vis.display(q) 
            time.sleep(5e-3) 
    
    return hist 

make_random_descent() 

# ---------------------------------- 

def compute_min_dist_env(q): 
    ''' 
    Compute the minimum distance between the robot and the obstacles. 
    '''

    pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)

    is_collision = pin.computeCollisions(robot.collision_model,robot.collision_data, False) 

    if is_collision: 
        return 0.0 

    # Compute the distance between each collision pair for a given GeometryModel and associated GeometryData.
    distance_idx = pin.computeDistances(robot.collision_model,robot.collision_data) 
    min_distance = robot.collision_data.distanceResults[distance_idx].min_distance 

    return min_distance 

# test function 
min_distance = compute_min_dist_env(robot.q0) 
print("Minimum distance to environment = ", min_distance) 




    
# ---------------------------------- 
# KEEP SCRIPT RUNNING 
# ---------------------------------- 

print("Keep Meshcat server alive")

# Keep the script running to keep the Meshcat server alive
while True:
    time.sleep(1)
    

    