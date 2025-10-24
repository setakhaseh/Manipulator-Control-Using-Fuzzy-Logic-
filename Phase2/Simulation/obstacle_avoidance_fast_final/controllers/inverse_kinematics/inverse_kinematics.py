import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import sys
import tempfile
import math
from controller import Supervisor

try:
    import ikpy
    from ikpy.chain import Chain
except ImportError:
    sys.exit('The "ikpy" Python module is not installed. '
             'To run this sample, please upgrade "pip" and install ikpy with this command: "pip install ikpy"')

if ikpy.__version__[0] < '3':
    sys.exit('The "ikpy" Python module version is too old. '
             'Please upgrade "ikpy" Python module to version "3.0" or newer with this command: "pip install --upgrade ikpy"')

# Function to calculate joint angles for a given position
def get_joint_angles(position):
    """Calculate the joint angles to reach the desired position."""
    initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]
    joint_angles = armChain.inverse_kinematics(position, max_iter=IKPY_MAX_ITERATIONS, initial_position=initial_position)
    return joint_angles[1:]  # Exclude the base angle

def get_recent_angles():
    """Retrieve the recent angles of the arm joints from position sensors."""
    return [sensor.getValue() for sensor in position_sensors]

# Define constants
IKPY_MAX_ITERATIONS = 4
MAX_LOOP_ITERATIONS = 1000

# Define the JointFuzzyPIDController class
class JointFuzzyController:
    def __init__(self, joint_name):
        self.e = ctrl.Antecedent(np.arange(-1.5, 1.5, 0.1), f'{joint_name}_e')
        self.ec = ctrl.Antecedent(np.arange(-1.5, 1.5, 0.1), f'{joint_name}_ec')
        self.velocity = ctrl.Consequent(np.arange(0, 2, 0.1), f'{joint_name}_velocity')
        self.velocityob = ctrl.Consequent(np.arange(0, 2, 0.1), f'{joint_name}_velocityob')
        self.myvelocity = ctrl.Antecedent(np.arange(0, 2, 0.1), f'{joint_name}_myvelocity')
        self.distance_obstacle = ctrl.Antecedent(np.arange(0, 21, 0.1), f'{joint_name}_distance_obstacle')
        self.distance_target = ctrl.Antecedent(np.arange(0, 21, 0.1), f'{joint_name}_distance_target')
        self._init_membership_functions()
        self._init_rules_velocity()
        self._init_rules_ob_velocity()

        # Initialize the control systems
        self.ctrl_vrules = ctrl.ControlSystem(self.vrules)
        self.sim_vrules = ctrl.ControlSystemSimulation(self.ctrl_vrules)
        
        self.ctrl_obvrules = ctrl.ControlSystem(self.obvrules)
        self.sim_obvrules = ctrl.ControlSystemSimulation(self.ctrl_obvrules)

    def _init_membership_functions(self):
        # Membership functions for error (e)
        self.e['NB'] = fuzz.trimf(self.e.universe, [-1.5, -1.5, -0.75])
        self.e['NS'] = fuzz.trimf(self.e.universe, [-1.5, 0, 0.75])
        self.e['ZO'] = fuzz.trimf(self.e.universe, [-0.75, 0, 0.75])
        self.e['PS'] = fuzz.trimf(self.e.universe, [0, 0.75, 1.5])
        self.e['PB'] = fuzz.trimf(self.e.universe, [0.75, 1.5, 1.5])

        # Membership functions for error change (ec)
        self.ec['NB'] = fuzz.trimf(self.ec.universe, [-1.5, -1.5, -0.75])
        self.ec['NS'] = fuzz.trimf(self.ec.universe, [-1.5, 0, 0.75])
        self.ec['ZO'] = fuzz.trimf(self.ec.universe, [-0.75, 0, 0.75])
        self.ec['PS'] = fuzz.trimf(self.ec.universe, [0, 0.75, 1.5])
        self.ec['PB'] = fuzz.trimf(self.ec.universe, [0.75, 1.5, 1.5])

        # Membership functions for velocity
        self.velocity['Slow'] = fuzz.trimf(self.velocity.universe, [0, 0, 1])
        self.velocity['Medium'] = fuzz.trimf(self.velocity.universe, [0.5, 1, 1.5])
        self.velocity['Fast'] = fuzz.trimf(self.velocity.universe, [1, 2, 2])
        # --------------------------------------Membership functions for distance with the obstacle---------------------------
        # Membership functions for distance with the obstacle       
        self.distance_obstacle['VN'] = fuzz.trimf(self.distance_obstacle.universe, [0, 0, 3])       # Very Near
        self.distance_obstacle['N'] = fuzz.trimf(self.distance_obstacle.universe, [2, 3, 5])       # Near
        self.distance_obstacle['F'] = fuzz.trapmf(self.distance_obstacle.universe, [4, 5, 10, 16])      # Far
        self.distance_obstacle['VF'] = fuzz.trapmf(self.distance_obstacle.universe, [15, 16, 20, 21])    # Very Far

        # Membership functions for distance with the target
        self.distance_target['VN'] = fuzz.trimf(self.distance_target.universe, [0, 0, 3])       # Very Near
        self.distance_target['N'] = fuzz.trimf(self.distance_target.universe, [2, 3, 5])       # Near
        self.distance_target['F'] = fuzz.trapmf(self.distance_target.universe, [4, 5, 10, 16])      # Far
        self.distance_target['VF'] = fuzz.trapmf(self.distance_target.universe, [15, 16, 20, 21])    # Very Far
        #Define velocity with obstacle
        # Define fuzzy membership functions for velocity
        self.velocityob['VS'] = fuzz.trimf(self.velocityob.universe, [0, 0, 0.5])     # Very Slow
        self.velocityob['MS'] = fuzz.trimf(self.velocityob.universe, [0, 0.5, 1])      # Middle Slow
        self.velocityob['S'] = fuzz.trimf(self.velocityob.universe, [0.5, 1, 1.5])     # Slow
        self.velocityob['M'] = fuzz.trimf(self.velocityob.universe, [1, 1.5, 2])       # Middle
        self.velocityob['F'] = fuzz.trimf(self.velocityob.universe, [1.5, 2, 2])       # Fast
        self.velocityob['MF'] = fuzz.trimf(self.velocityob.universe, [1.75, 2, 2])     # Middle Fast
        self.velocityob['VF'] = fuzz.trimf(self.velocityob.universe, [2, 2, 2])        # Very Fast 

        # my velocity
        self.myvelocity['VS'] = fuzz.trimf(self.myvelocity.universe, [0, 0, 0.5])     # Very Slow
        self.myvelocity['MS'] = fuzz.trimf(self.myvelocity.universe, [0, 0.5, 1])      # Middle Slow
        self.myvelocity['S'] = fuzz.trimf(self.myvelocity.universe, [0.5, 1, 1.5])     # Slow
        self.myvelocity['M'] = fuzz.trimf(self.myvelocity.universe, [1, 1.5, 2])       # Middle
        self.myvelocity['F'] = fuzz.trimf(self.myvelocity.universe, [1.5, 2, 2])       # Fast
        self.myvelocity['MF'] = fuzz.trimf(self.myvelocity.universe, [1.75, 2, 2])     # Middle Fast
        self.myvelocity['VF'] = fuzz.trimf(self.myvelocity.universe, [2, 2, 2])        # Very Fast

    def _init_rules_velocity(self):
        # Define rules for velocity control
        self.vrules = [
            ctrl.Rule(self.e['NB'] & self.ec['NB'], self.velocity['Slow']),
            ctrl.Rule(self.e['NB'] & self.ec['NS'], self.velocity['Slow']),
            ctrl.Rule(self.e['NB'] & self.ec['ZO'], self.velocity['Medium']),
            ctrl.Rule(self.e['NB'] & self.ec['PS'], self.velocity['Medium']),
            ctrl.Rule(self.e['NB'] & self.ec['PB'], self.velocity['Fast']),

            ctrl.Rule(self.e['ZO'] & self.ec['NB'], self.velocity['Medium']),
            ctrl.Rule(self.e['ZO'] & self.ec['NS'], self.velocity['Medium']),
            ctrl.Rule(self.e['ZO'] & self.ec['ZO'], self.velocity['Medium']),
            ctrl.Rule(self.e['ZO'] & self.ec['PS'], self.velocity['Medium']),
            ctrl.Rule(self.e['ZO'] & self.ec['PB'], self.velocity['Fast']),

            ctrl.Rule(self.e['PB'] & self.ec['NB'], self.velocity['Medium']),
            ctrl.Rule(self.e['PB'] & self.ec['NS'], self.velocity['Medium']),
            ctrl.Rule(self.e['PB'] & self.ec['ZO'], self.velocity['Fast']),
            ctrl.Rule(self.e['PB'] & self.ec['PS'], self.velocity['Fast']),
            ctrl.Rule(self.e['PB'] & self.ec['PB'], self.velocity['Fast']),
        ]
    def _init_rules_ob_velocity(self):
        self.obvrules = [
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['VN'] & self.myvelocity['VS'], self.velocityob['VS']),
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['VN'] & self.myvelocity['MS'], self.velocityob['VS']),
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['VN'] & self.myvelocity['S'], self.velocityob['VS']),
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['VN'] & self.myvelocity['M'], self.velocityob['MS']),
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['VN'] & self.myvelocity['F'], self.velocityob['M']),
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['VN'] & self.myvelocity['MF'], self.velocityob['M']),
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['VN'] & self.myvelocity['VF'], self.velocityob['F']),
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['N'] & self.myvelocity['VS'], self.velocityob['VS']),
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['N'] & self.myvelocity['MS'], self.velocityob['VS']),
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['N'] & self.myvelocity['S'], self.velocityob['MS']),
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['N'] & self.myvelocity['M'], self.velocityob['S']),
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['N'] & self.myvelocity['F'], self.velocityob['M']),
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['N'] & self.myvelocity['MF'], self.velocityob['F']),
            ctrl.Rule(self.distance_target['VN'] & self.distance_obstacle['N'] & self.myvelocity['VF'], self.velocityob['MF']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['N'] & self.myvelocity['VS'], self.velocityob['VS']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['N'] & self.myvelocity['MS'], self.velocityob['VS']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['N'] & self.myvelocity['S'], self.velocityob['MS']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['N'] & self.myvelocity['M'], self.velocityob['S']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['N'] & self.myvelocity['F'], self.velocityob['M']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['N'] & self.myvelocity['MF'], self.velocityob['F']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['N'] & self.myvelocity['VF'], self.velocityob['MF']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['F'] & self.myvelocity['VS'], self.velocityob['VS']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['F'] & self.myvelocity['MS'], self.velocityob['VS']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['F'] & self.myvelocity['S'], self.velocityob['MS']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['F'] & self.myvelocity['M'], self.velocityob['S']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['F'] & self.myvelocity['F'], self.velocityob['M']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['F'] & self.myvelocity['MF'], self.velocityob['F']),
            ctrl.Rule(self.distance_target['N'] & self.distance_obstacle['F'] & self.myvelocity['VF'], self.velocityob['MF']),
        ]


    def determine_best_distance_category(self, distance_value):
    # Calculate the membership values for each distance category
        distance_membership_vf = fuzz.interp_membership(self.distance_obstacle.universe, self.distance_obstacle['VF'].mf, distance_value)
        distance_membership_f = fuzz.interp_membership(self.distance_obstacle.universe, self.distance_obstacle['F'].mf, distance_value)
        distance_membership_n = fuzz.interp_membership(self.distance_obstacle.universe, self.distance_obstacle['N'].mf, distance_value)
        distance_membership_vn = fuzz.interp_membership(self.distance_obstacle.universe, self.distance_obstacle['VN'].mf, distance_value)

    # Create a dictionary to store membership values
        membership_values = {
            'VF': distance_membership_vf,
            'F': distance_membership_f,
            'N': distance_membership_n,
            'VN': distance_membership_vn
        }

    # Determine the category with the highest membership value
        best_category = max(membership_values, key=membership_values.get)

        return best_category

    def compute_velocity_ob(self, error, error_change, myvelocity, distance_target_v, distance_obstacle_value):
    # Determine the best distance categories for obstacle and target
        best_distance_obstacle = self.determine_best_distance_category(distance_obstacle_value)
        best_distance_target = self.determine_best_distance_category(distance_target_v)

    # Check conditions to determine which rules to use or return 20 to change the path
        if best_distance_obstacle == 'VF':
            self.sim_vrules.input[f'{self.e.label}'] = error
            self.sim_vrules.input[f'{self.ec.label}'] = error_change
            self.sim_vrules.compute()
            return self.sim_vrules.output[f'{self.velocity.label}']
        elif best_distance_obstacle == 'F' and (best_distance_target != 'N'):
            self.sim_vrules.input[f'{self.e.label}'] = error
            self.sim_vrules.input[f'{self.ec.label}'] = error_change
            self.sim_vrules.compute()
            return self.sim_vrules.output[f'{self.velocity.label}']
        elif best_distance_obstacle == 'N' and (best_distance_target == 'F' or best_distance_target=='VF'):
            return 20
        elif best_distance_obstacle == 'VN' and (best_distance_obstacle != 'VN'):
            return 20
        else:
            self.sim_obvrules.input[f'{self.distance_target.label}'] = distance_target_v
            self.sim_obvrules.input[f'{self.distance_obstacle.label}'] = distance_obstacle_value
            self.sim_obvrules.input[f'{self.myvelocity.label}'] = myvelocity
            self.sim_obvrules.compute()
            return self.sim_obvrules.output[f'{self.velocityob.label}']

    def compute_velocity(self, error, error_change):
        self.sim_vrules.input[f'{self.e.label}'] = error
        self.sim_vrules.input[f'{self.ec.label}'] = error_change
        self.sim_vrules.compute()
        return self.sim_vrules.output[f'{self.velocity.label}']

# Initialize the Webots Supervisor.
supervisor = Supervisor()
timeStep = int(4 * supervisor.getBasicTimeStep())

# Get target and obstacle nodes
target = supervisor.getFromDef('TARGET')
if target:
    target_position = target.getPosition()
    print("Target position:", target_position)
else:
    sys.exit("Target node not found in the simulation.")
    
obstacle = supervisor.getFromDef('OBSTACLE')
if obstacle:
    obstacle_position = obstacle.getPosition()
    print("Obstacle position:", obstacle_position)
else:
    sys.exit("Obstacle node not found in the simulation.")

# Create the arm chain from the URDF
filename = None
with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as file:
    filename = file.name
    file.write(supervisor.getUrdf().encode('utf-8'))
armChain = Chain.from_urdf_file(filename, active_links_mask=[False, True, True, True, True, True, True, False])

# Initialize the arm motors and encoders.
motors = []
position_sensors = []
for link in armChain.links:
    if 'motor' in link.name:
        motor = supervisor.getDevice(link.name)
        motor.setVelocity(1.0)
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timeStep)
        motors.append(motor)
        position_sensors.append(position_sensor)

# Initialize fuzzy controllers for each active link
fuzzy_controllers = [JointFuzzyController(joint_name=f'joint_{i}') for i in range(len(armChain.active_links_mask))]

# Loop variables
elapsed_time = 0
loop_counter = 0
target_position = [0.5, 0.5, 0.2]
previous_errors = [0] * len(motors)
integral_of_error = [0] * len(motors)

print('Moving to target position...')

def determine_side(target_position, obstacle_position):
    """
    Determine which side of the target the obstacle is on.

    Parameters:
    target (tuple): The (x, y, z) coordinates of the target.
    obstacle (tuple): The (x, y, z) coordinates of the obstacle.

    Returns:
    str: The side of the target the obstacle is on ('right', 'left', 'up', 'down', 'front', 'back').
    """
    target_x, target_y, target_z = target_position
    obstacle_x, obstacle_y, obstacle_z = obstacle_position

    # Determine the relative position
    delta_x = obstacle_x - target_x
    delta_y = obstacle_y - target_y
    delta_z = obstacle_z - target_z

    # Define thresholds for determining sides
    threshold = 0.001  # A small threshold to handle floating point precision

    if abs(delta_x) > abs(delta_y) and abs(delta_x) > abs(delta_z):
        if delta_x > threshold:
            return "right"
        elif delta_x < -threshold:
            return "left"
    elif abs(delta_y) > abs(delta_x) and abs(delta_y) > abs(delta_z):
        if delta_y > threshold:
            return "up"
        elif delta_y < -threshold:
            return "down"
    else:
        if delta_z > threshold:
            return "front"
        elif delta_z < -threshold:
            return "back"

    return "same position"

def calculate_positions(target_position, obstacle_position):
    """
    Calculate four specific locations based on the relative position of the obstacle to the target.

    Parameters:
    target_position (tuple): The (x, y, z) coordinates of the target.
    obstacle_position (tuple): The (x, y, z) coordinates of the obstacle.

    Returns:
    tuple: The four calculated positions (A, B, C, target).
    """
    target_x, target_y, target_z = target_position  # Unpack target position
    obstacle_x, obstacle_y, obstacle_z = obstacle_position  # Unpack obstacle position

    # Calculate distance based on the relative position
    distance_x = abs(obstacle_x - target_x)
    distance_y = abs(obstacle_y - target_y)
    distance_z = abs(obstacle_z - target_z)

    # Determine the side based on the obstacle's position relative to the target
    if obstacle_x < target_x:
        side = "left"
    elif obstacle_x > target_x:
        side = "right"
    elif obstacle_y < target_y:
        side = "down"
    elif obstacle_y > target_y:
        side = "up"
    elif obstacle_z < target_z:
        side = "back"
    else:
        side = "front"

    # Calculate positions A, B, C based on the determined side
    if side == "left":
        A = [obstacle_x - distance_x, obstacle_y, obstacle_z]
        B = [A[0], A[1] + distance_x, A[2]]
        C = [target_x, target_y + distance_x, target_z]
    elif side == "right":
        A = [obstacle_x + distance_x, obstacle_y, obstacle_z]
        B = [A[0], A[1] + distance_x, A[2]]
        C = [target_x, target_y + distance_x, target_z]
    elif side == "up":
        A = [obstacle_x, obstacle_y + distance_y, obstacle_z]
        B = [A[0] + distance_y, A[1], A[2]]
        C = [target_x + distance_y, target_y, target_z]
    elif side == "down":
        A = [obstacle_x, obstacle_y - distance_y, obstacle_z]
        B = [A[0] + distance_y, A[1], A[2]]
        C = [target_x + distance_y, target_y, target_z]
    elif side == "front":
        A = [obstacle_x, obstacle_y, obstacle_z + distance_z]
        B = [A[0] + distance_z, A[1], A[2]]
        C = [target_x + distance_z, target_y, target_z]
    elif side == "back":
        A = [obstacle_x, obstacle_y, obstacle_z - distance_z]
        B = [A[0] + distance_z, A[1], A[2]]
        C = [target_x + distance_z, target_y, target_z]

    return A, B, C, target_position  # Return the calculated positions




def get_recent_velocity(motors):
    """Retrieve a single recent velocity value from the motors."""
    velocities = [motor.getVelocity() for motor in motors]
    # Use an aggregate, e.g., the average or the first motor's velocity
    return float(sum(velocities))


 
 

# Main loop
while supervisor.step(timeStep) != -1:
    elapsed_time += timeStep / 1000.0  # Convert timeStep to seconds
    loop_counter += 1    
    
    # Calculate joint angles for the target position
    joint_angles_target = get_joint_angles(target_position)
    
    # Calculate joint angles for the obstacle position
    joint_angles_obstacle = get_joint_angles(obstacle_position)
    
    # Debugging information
    print(f"Number of Motors: {len(motors)}, Number of Joint Angles: {len(joint_angles_target)}")
    
    # Ensure we only set positions for available motors (for target)
    for i in range(min(len(motors), len(joint_angles_target))):
        motors[i].setPosition(joint_angles_target[i])

    # Get the recent joint angles
    recent_angles = get_recent_angles()
    
    # Compute the sum of absolute differences for the target
    distance_target_v = sum(
        abs(joint_angles_target[i] - recent_angles[i]) 
        for i in range(min(len(joint_angles_target), len(recent_angles)))
    )

    #Compute the sum of absolute differences for the obstacle
    distance_obstacle_value = sum(
        abs(joint_angles_obstacle[i] - recent_angles[i]) 
        for i in range(min(len(joint_angles_obstacle), len(recent_angles)))
    )
    
    # Command motors using fuzzy controller
    for i in range(min(len(motors), len(joint_angles_target))):
        current_angle = position_sensors[i].getValue()
        
        # Print recent motor velocities
        myvelocity = get_recent_velocity(motors)
        # print("Recent Motor Velocities:", myvelocity)
        
        # Calculate error and change in error
        error = joint_angles_target[i] - current_angle
        error_change = error - previous_errors[i]
        
        # Compute velocity using fuzzy controller
        velocity = fuzzy_controllers[i].compute_velocity_ob(error, error_change, myvelocity, distance_target_v, distance_obstacle_value)
        
        if velocity == 20.0:
       
        # Calculate positions A, B, C, and target only when velocity is 20
            A, B, C, target_position = calculate_positions(target_position, obstacle_position)
            print("Positions calculated: A =", A, ", B =", B, ", C =", C, ", Target =", target_position)
            
            previous_errors = [0] * len(motors)
            integral_of_error = [0] * len(motors)

            
            motors[i].setVelocity(0)  # Stop the motor
            # Calculate joint angles for the target position
            joint_angles_A = get_joint_angles(A)
            # Get the recent joint angles
            recent_angles = get_recent_angles()
            # Move to position A
            # Command motors using fuzzy controller
            for i in range(min(len(motors), len(joint_angles_A))):
                current_angle = position_sensors[i].getValue()
        
                # Calculate error and change in error
                error = joint_angles_A[i] - current_angle
                error_change = error - previous_errors[i]
        
                # Compute velocity using fuzzy controller
                velocity = fuzzy_controllers[i].compute_velocity( error, error_change)

                # Update motor velocity and position
                motors[i].setVelocity(velocity)
                motors[i].setPosition(joint_angles_A[i])

                # Store previous error and update integral of error
                previous_errors[i] = error
                integral_of_error[i] += error * (timeStep / 1000.0)
                if all(abs(joint_angles_A[i] - recent_angles[i]) < 0.01 for i in range(min(len(joint_angles_A), len(recent_angles)))):
                    print("Target position AAAAAA reached. Stopping simulation.")
                
                    break

                
                
            previous_errors = [0] * len(motors)
            integral_of_error = [0] * len(motors)

            
            motors[i].setVelocity(0)  # Stop the motor
            # Calculate joint angles for the target position
            joint_angles_B = get_joint_angles(B)
            # Get the recent joint angles
            recent_angles = get_recent_angles()
            # Move to position A
            # Command motors using fuzzy controller
            for i in range(min(len(motors), len(joint_angles_B))):
                current_angle = position_sensors[i].getValue()
        
                # Calculate error and change in error
                error = joint_angles_B[i] - current_angle
                error_change = error - previous_errors[i]
        
                # Compute velocity using fuzzy controller
                velocity = fuzzy_controllers[i].compute_velocity( error, error_change)

                # Update motor velocity and position
                motors[i].setVelocity(velocity)
                motors[i].setPosition(joint_angles_B[i])

                # Store previous error and update integral of error
                previous_errors[i] = error
                integral_of_error[i] += error * (timeStep / 1000.0)
                if all(abs(joint_angles_B[i] - recent_angles[i]) < 0.01 for i in range(min(len(joint_angles_B), len(recent_angles)))):
                    print("Target position BBBBBB reached. Stopping simulation.")
                
                    break
                
                
                    
                
                
            previous_errors = [0] * len(motors)
            integral_of_error = [0] * len(motors)

            
            motors[i].setVelocity(0)  # Stop the motor
            # Calculate joint angles for the target position
            joint_angles_C = get_joint_angles(C)
            # Get the recent joint angles
            recent_angles = get_recent_angles()
            # Move to position A
            # Command motors using fuzzy controller
            for i in range(min(len(motors), len(joint_angles_C))):
                current_angle = position_sensors[i].getValue()
        
                # Calculate error and change in error
                error = joint_angles_C[i] - current_angle
                error_change = error - previous_errors[i]
        
                # Compute velocity using fuzzy controller
                velocity = fuzzy_controllers[i].compute_velocity( error, error_change)

                # Update motor velocity and position
                motors[i].setVelocity(velocity)
                motors[i].setPosition(joint_angles_C[i])

                # Store previous error and update integral of error
                previous_errors[i] = error
                integral_of_error[i] += error * (timeStep / 1000.0)
                
                if all(abs(joint_angles_C[i] - recent_angles[i]) < 0.01 for i in range(min(len(joint_angles_C), len(recent_angles)))):
                    print("Target position CCCCCCC reached. Stopping simulation.")   
                
                    break
                
                 
                    
            
            previous_errors = [0] * len(motors)
            integral_of_error = [0] * len(motors)

            
            motors[i].setVelocity(0)  # Stop the motor
            # Calculate joint angles for the target position
            joint_angles_target_position = get_joint_angles(target_position)
            # Get the recent joint angles
            recent_angles = get_recent_angles()
            for i in range(min(len(motors), len(joint_angles_target_position))):
                current_angle = position_sensors[i].getValue()
        
                # Calculate error and change in error
                error = joint_angles_target[i] - current_angle
                error_change = error - previous_errors[i]
        
                # Compute velocity using fuzzy controller
                velocity = fuzzy_controllers[i].compute_velocity( error, error_change)

                # Update motor velocity and position
                motors[i].setVelocity(velocity)
                motors[i].setPosition(joint_angles_target[i])

                # Store previous error and update integral of error
                previous_errors[i] = error
                integral_of_error[i] += error * (timeStep / 1000.0)
                if all(abs(joint_angles_target_position[i] - recent_angles[i]) < 0.01 for i in range(min(len(joint_angles_target_position), len(recent_angles)))):
                    print("Target position reached. Stopping simulation.")   
                    break
            
        
        
        else:
            # Update motor velocity and position
            motors[i].setVelocity(velocity)
            motors[i].setPosition(joint_angles_target[i])

            # Print recent motor velocities
            myvelocity = get_recent_velocity(motors)
           # print("Recent Motor Velocities:", myvelocity)
        
            # Store previous error and update integral of error
            previous_errors[i] = error
            integral_of_error[i] += error * (timeStep / 1000.0)
        
        
        
    
            
                

        
    
    # Check if the elapsed time is a whole number
    if int(elapsed_time) % 1 == 0: 
        recent_angles = get_recent_angles()
        print("Recent Joint Angles at t =", int(elapsed_time), "s:", recent_angles)
        
        # Print the calculated sums
        print("Sum of absolute differences (Target) at t =", int(elapsed_time), "s:", distance_target_v)
        print("Sum of absolute differences (Obstacle) at t =", int(elapsed_time), "s:", distance_obstacle_value)

    # Optionally, add a condition to stop after reaching the target or avoiding the obstacle
    if all(abs(joint_angles_target[i] - recent_angles[i]) < 0.01 for i in range(min(len(joint_angles_target), len(recent_angles)))):
        print("Target position reached.")
        break
