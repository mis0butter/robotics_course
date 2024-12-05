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
dist_target = compute_distance_target(robot.q0) 
print("Distance to target = ", dist_target) 

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
            time.sleep(1) 
    
    return hist 

# test function 
hist = make_random_descent() 
print("History of configurations = ", hist) 

# ---------------------------------- 
# CONFIGURATION SPACE 
# ---------------------------------- 

def find_min_distance_collision(q): 
    ''' 
    Compute the minimum distance between the robot and the obstacles. 
    '''

    pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)

    is_collision = pin.computeCollisions(robot.collision_model,robot.collision_data, False) 

    if is_collision: 
        return 0.0 

    # Compute distance between each collision pair for a given GeometryModel and associated GeometryData.
    distance_idx = pin.computeDistances(robot.collision_model,robot.collision_data) 
    min_distance = robot.collision_data.distanceResults[distance_idx].min_distance 

    return min_distance 

# test function 
min_distance = find_min_distance_collision(robot.q0) 
print("Minimum distance to environment = ", min_distance) 

# ---------------------------------- 

def sample_config_space(num_samples=500):
    ''' 
    Sample num_samples configurations and store in two lists: 
        - hist_coll - if config is in collision 
        - hist_free - if config is in free space 
    along with distance to the target and distance to obstacles. 
    '''
    
    hist_coll = [] 
    hist_free = [] 
    
    for i in range(num_samples): 
        
        q            = sample_valid_config(check = False) 
        dist_target  = compute_distance_target(q) 
        is_collision = check_collision(q) 
        
        if is_collision:             
            hist_coll.append( list(q.flat) + [ dist_target, 1e-2 ] ) 
        else: 
            dist_coll = find_min_distance_collision(q) 
            hist_free.append( list(q.flat) + [ dist_target, dist_coll ] ) 
    
    return hist_coll, hist_free 

# ---------------------------------- 

def plot_config_space(hist_coll, hist_free, marker_size = 20): 
    '''
    Plot 2 "scatter" plots: the first one plot the distance to the target for 
    each configuration, the second plots the distance to the obstacles (axis q1,q2, 
    distance in the color space).
    ''' 
    
    hist_total = hist_coll + hist_free 
    h          = np.array(hist_total)      
    
    fig, axs = plt.subplots(2, 1 , figsize=(10, 8))
    
    axs[0].scatter(h[:,0], h[:,1], c=h[:,2], s=marker_size, lw=0)
    axs[0].set_title("Distance to the target")
    fig.colorbar(axs[0].collections[0], ax=axs[0])
    
    axs[1].scatter(h[:,0], h[:,1], c=h[:,3], s=marker_size, lw=0)
    axs[1].set_title("Distance to the obstacles")
    fig.colorbar(axs[1].collections[0], ax=axs[1])
     
    # plt.tight_layout()
    
    return fig, axs

# test 
hist_coll, hist_free = sample_config_space(num_samples = 5000) 
fig, axs = plot_config_space(hist_coll, hist_free)

# Example of augmenting the plot
axs[0].set_xlabel('Joint 1')
axs[0].set_ylabel('Joint 2')
axs[1].set_xlabel('Joint 1')
axs[1].set_ylabel('Joint 2')

print("Plot of configuration space displayed") 
# plt.show()
print("Plot updated") 

# test 
hist_coll, hist_free = sample_config_space(num_samples = 5000) 
plot_config_space(hist_coll, hist_free) 

# ---------------------------------- 
# Display feasible trajectory by random walk 
# ---------------------------------- 
 
q_init = np.array( [-1.1, -3.] ) 

for i in range(100): 
    
    traj        = make_random_descent(q_init) 
    dist_target = compute_distance_target(traj[-1]) 
    
    if dist_target < 5e-2: 
        print('Found feasible trajectory') 
        break 

traj = np.array(traj) 

# choose trajectory end to be in [pi, -pi] through angle normalization: 
#   traj[-1]    - Gets the last element of the trajectory array
#   + np.pi     - Shifts the angle by +π
#   % (2*np.pi) - Uses modulo to wrap the angle into [0, 2π]
#   - np.pi     - Shifts back by -π to get into [-π, π] range
q_end = ( traj[-1] + np.pi ) % ( 2*np.pi ) - np.pi 
traj += ( q_end - traj[-1] )   

# plt.ion()  # Turn on interactive mode
fig, axs = plot_config_space(hist_coll, hist_free) 
axs[0].plot( traj[:,0], traj[:,1], color = 'black', lw = 2 ) 
plt.show(block=False)
    
# ---------------------------------- 
# KEEP SCRIPT RUNNING 
# ---------------------------------- 

print("Keep Meshcat server alive")

# Keep the script running to keep the Meshcat server alive
while True:
    time.sleep(1)
    



    