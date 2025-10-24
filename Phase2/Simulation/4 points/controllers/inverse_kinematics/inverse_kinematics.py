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
        self.distance_obstacle['VN'] = fuzz.trimf(self.distance_obstacle.universe, [0, 0, 5])       # Very Near
        self.distance_obstacle['N'] = fuzz.trimf(self.distance_obstacle.universe, [0, 5, 10])       # Near
        self.distance_obstacle['F'] = fuzz.trimf(self.distance_obstacle.universe, [5, 10, 15])      # Far
        self.distance_obstacle['VF'] = fuzz.trimf(self.distance_obstacle.universe, [10, 15, 21])    # Very Far

        # Membership functions for distance with the target
        self.distance_target['VN'] = fuzz.trimf(self.distance_target.universe, [0, 0, 5])       # Very Near
        self.distance_target['N'] = fuzz.trimf(self.distance_target.universe, [0, 5, 10])       # Near
        self.distance_target['F'] = fuzz.trimf(self.distance_target.universe, [5, 10, 15])      # Far
        self.distance_target['VF'] = fuzz.trimf(self.distance_target.universe, [10, 15, 21])    # Very Far
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


    def compute_velocity_ob(self, error, error_change, myvelocity, distance_target_v, distance_obstacle_value):
    # Determine the membership values for distance_obstacle and absolute error
        distance_membership_vf = fuzz.interp_membership(self.distance_obstacle.universe, self.distance_obstacle['VF'].mf, distance_obstacle_value)
        distance_membership_f = fuzz.interp_membership(self.distance_obstacle.universe, self.distance_obstacle['F'].mf, distance_obstacle_value)
        distance_membership_vn = fuzz.interp_membership(self.distance_obstacle.universe, self.distance_obstacle['VN'].mf, distance_obstacle_value)
        distance_membership_n = fuzz.interp_membership(self.distance_obstacle.universe, self.distance_obstacle['N'].mf, distance_obstacle_value)
    
        distance_target_membership_vn = fuzz.interp_membership(self.distance_target.universe, self.distance_target['VN'].mf, distance_target_v)
        distance_target_membership_n = fuzz.interp_membership(self.distance_target.universe, self.distance_target['N'].mf, distance_target_v)
        distance_target_membership_f = fuzz.interp_membership(self.distance_target.universe, self.distance_target['F'].mf, distance_target_v)
        distance_target_membership_vf = fuzz.interp_membership(self.distance_target.universe, self.distance_target['VF'].mf, distance_target_v)

    # Check conditions to determine which rules to use or return 20
        if distance_membership_vf > 0:
            self.sim_vrules.input[f'{self.e.label}'] = error
            self.sim_vrules.input[f'{self.ec.label}'] = error_change
            self.sim_vrules.compute()
            return self.sim_vrules.output[f'{self.velocity.label}']
        elif distance_membership_f > 0 and (distance_target_membership_vn > 0 or distance_target_membership_f > 0 or distance_target_membership_vf > 0):
            self.sim_vrules.input[f'{self.e.label}'] = error
            self.sim_vrules.input[f'{self.ec.label}'] = error_change
            self.sim_vrules.compute()
            return self.sim_vrules.output[f'{self.velocity.label}']
        elif distance_target_membership_n > 0 and distance_membership_vn > 0:
            return 20
        elif distance_membership_f > 0 and (distance_membership_vn > 0 or distance_membership_n > 0):
            return 20
        elif distance_target_membership_vf > 0 and (distance_membership_vn > 0 or distance_membership_n > 0):
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

# Get target and obstacle positions
target_position = [0.5, 0.5, 0.2]  # Example target position (x, y, z)
obstacle_position = [0.3, 0.8, 0.7]

# Print target and obstacle positions
print("Target position:", target_position)    
print("Obstacle position:", obstacle_position)  

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
    target (tuple): The (x, y, z) coordinates of the target.
    obstacle (tuple): The (x, y, z) coordinates of the obstacle.

    Returns:
    tuple: The four calculated positions (A, B, C, target).
    """
    side = determine_side(target_position, obstacle_position)
    target_x, target_y, target_z = target_position
    obstacle_x, obstacle_y, obstacle_z = obstacle_position

    if side == "left":
        distance=abs((obstacle_x - target_x))
        # Position A: on the left side of the obstacle with 2*distance
        A_x = obstacle_x - 2 * distance
        A_y = obstacle_y
        A_z = obstacle_z

        # Position B: on the upside of A with 2*distance
        B_x = A_x
        B_y = A_y + 2 * distance
        B_z = A_z

        # Position C: on the upside of the target with 2*distance
        C_x = target_x
        C_y = target_y + 2 * distance
        C_z = target_z

        # Target position
        target_position = (target_x, target_y, target_z)
    if side == "right":
        distance=abs((obstacle_x - target_x))
        # Position A: on the left side of the obstacle with 2*distance
        A_x = obstacle_x + 2 * distance
        A_y = obstacle_y
        A_z = obstacle_z

        # Position B: on the upside of A with 2*distance
        B_x = A_x
        B_y = A_y + 2 * distance
        B_z = A_z

        # Position C: on the upside of the target with 2*distance
        C_x = target_x
        C_y = target_y + 2 * distance
        C_z = target_z

        # Target position
        target_position = (target_x, target_y, target_z)
    if side == "up":
        distance=abs((obstacle_y - target_y))
        # Position A: on the left side of the obstacle with 2*distance
        A_x = obstacle_x 
        A_y = obstacle_y + 2 * distance
        A_z = obstacle_z

        # Position B: on the upside of A with 2*distance
        B_x = A_x + 2 * distance
        B_y = A_y 
        B_z = A_z

        # Position C: on the upside of the target with 2*distance
        C_x = target_x + 2 * distance
        C_y = target_y 
        C_z = target_z

        # Target position
        target_position = (target_x, target_y, target_z)
    if side == "down":
        distance=abs((obstacle_y - target_y))
        # Position A: on the left side of the obstacle with 2*distance
        A_x = obstacle_x 
        A_y = obstacle_y - 2 * distance
        A_z = obstacle_z

        # Position B: on the upside of A with 2*distance
        B_x = A_x + 2 * distance
        B_y = A_y 
        B_z = A_z

        # Position C: on the upside of the target with 2*distance
        C_x = target_x + 2 * distance
        C_y = target_y 
        C_z = target_z

        # Target position
        target_position = (target_x, target_y, target_z)
    if side == "front":
        distance=abs((obstacle_z - target_z))
        # Position A: on the left side of the obstacle with 2*distance
        A_x = obstacle_x 
        A_y = obstacle_y
        A_z = obstacle_z + 2 * distance

        # Position B: on the upside of A with 2*distance
        B_x = A_x+ 2 * distance
        B_y = A_y 
        B_z = A_z

        # Position C: on the upside of the target with 2*distance
        C_x = target_x + 2 * distance
        C_y = target_y 
        C_z = target_z

        # Target position
        target_position = (target_x, target_y, target_z)
    if side == "back":
        distance=abs((obstacle_z - target_z))
        # Position A: on the left side of the obstacle with 2*distance
        A_x = obstacle_x 
        A_y = obstacle_y
        A_z = obstacle_z - 2 * distance

        # Position B: on the upside of A with 2*distance
        B_x = A_x + 2 * distance
        B_y = A_y 
        B_z = A_z

        # Position C: on the upside of the target with 2*distance
        C_x = target_x + 2 * distance
        C_y = target_y 
        C_z = target_z

        # Target position
        target_position = (target_x, target_y, target_z)
    return ((A_x, A_y, A_z), (B_x, B_y, B_z), (C_x, C_y, C_z), target_position)

# Function to move the robotic arm to sequential positions
def move_to_positions(positions):
    """
    Move the robotic arm to the specified positions sequentially.
    Parameters:
    positions (list): A list of (x, y, z) positions to move to sequentially.
    """
    for position in positions:
        joint_angles = get_joint_angles(position)
        while True:
            # Set motor positions
            for i in range(min(len(motors), len(joint_angles))):
                motors[i].setPosition(joint_angles[i])
                motors[i].setVelocity(velocity)
                
            # Get the recent joint angles
            recent_angles = get_recent_angles()

            # Calculate the distance to the target position
            distance = sum(
                abs(joint_angles[i] - recent_angles[i])
                for i in range(min(len(joint_angles), len(recent_angles)))
            )

            # Break if the position is reached
            if distance < 0.01:  # Threshold for precision
                print(f"Position {position} reached.")
                break

            # Step simulation
            if supervisor.step(timeStep) == -1:
                break
# Assuming 'motors' is a list of your initialized motor devices
def get_recent_velocities(motors):
    """Retrieve the recent velocities of the motors."""
    return [motor.getVelocity() for motor in motors]
    
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
        myvelocity = get_recent_velocities(motors)
        print("Recent Motor Velocities:", myvelocity)
        
        # Calculate error and change in error
        error = joint_angles_target[i] - current_angle
        error_change = error - previous_errors[i]
        
        # Compute velocity using fuzzy controller
        velocity = fuzzy_controllers[i].compute_velocity_ob(error, error_change, myvelocity, distance_target_v, distance_obstacle_value)

        # Update motor velocity and position
        motors[i].setVelocity(velocity)
        motors[i].setPosition(joint_angles_target[i])
        
        # Store previous error and update integral of error
        previous_errors[i] = error
        integral_of_error[i] += error * (timeStep / 1000.0)
        
    if velocity == 20.0:
        # Calculate positions A, B, C, and target only when velocity is 20
        A, B, C, target_position = calculate_positions(target_position, obstacle_position)
        print("Positions calculated: A =", A, ", B =", B, ", C =", C, ", Target =", target_position)
        velocity = fuzzy_controllers[i].compute_velocity(error, error_change)
        error = joint_angles_target[i] - current_angle
        error_change = error - previous_errors[i]
        previous_errors[i] = error
        integral_of_error[i] += error * (timeStep / 1000.0)
                        
        # Move sequentially to positions A, B, C, and the target
        move_to_positions([A, B, C, target_position])

        # Exit the loop once the target position is reached
        print("Target position reached. Stopping simulation.")
        break
    else:
        print("Velocity is not 20. Skipping position calculations.")
    
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
