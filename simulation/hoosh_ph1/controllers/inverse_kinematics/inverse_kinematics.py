import numpy as np
import sys
import tempfile
import math
from controller import Supervisor
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import ikpy
from ikpy.chain import Chain

# Check for ikpy installation and version
try:
    if ikpy.__version__[0] < '3':
        raise ImportError
except ImportError:
    sys.exit('The "ikpy" Python module is not installed or is too old. '
             'Please upgrade "ikpy" with: "pip install --upgrade ikpy"')

# Define constants
IKPY_MAX_ITERATIONS = 4
MAX_LOOP_ITERATIONS = 1000

# Function to get joint angles
def get_joint_angles(target_position):
    """Calculate the joint angles to reach the desired target position."""
    initial_position = [0] * len(armChain.active_links_mask)
    print(f"Initial Position: {initial_position}")
    print(f"Target Position: {target_position}")
    
    try:
        joint_angles = armChain.inverse_kinematics(target_position, max_iter=IKPY_MAX_ITERATIONS, initial_position=initial_position)
        return joint_angles[1:]  # Exclude the base angle
    except Exception as e:
        print(f"Error in inverse kinematics: {e}")
        return initial_position  # Return initial position on failure

def get_recent_angles():
    """Retrieve the recent angles of the arm joints from position sensors."""
    return [sensor.getValue() for sensor in position_sensors]

IKPY_MAX_ITERATIONS = 4
MAX_LOOP_ITERATIONS = 100  # Set a maximum number of iterations for the main loop

# Define the JointFuzzyPIDController class
class JointFuzzyController:
    def __init__(self, joint_name):
        self.e = ctrl.Antecedent(np.arange(-1.5, 1.5, 0.1), f'{joint_name}_e')
        self.ec = ctrl.Antecedent(np.arange(-1.5, 1.5, 0.1), f'{joint_name}_ec')
        self.velocity = ctrl.Consequent(np.arange(0, 2, 0.1), f'{joint_name}_velocity')

        self._init_membership_functions()
        self._init_rules()

        self.ctrl = ctrl.ControlSystem(self.rules)
        self.sim = ctrl.ControlSystemSimulation(self.ctrl)

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

    def _init_rules(self):
        # Define rules for velocity control
        self.rules = [
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

    def compute_velocity(self, error, error_change):
        self.sim.input[f'{self.e.label}'] = error
        self.sim.input[f'{self.ec.label}'] = error_change
        self.sim.compute()
        return self.sim.output[f'{self.velocity.label}']

# Initialize the Webots Supervisor
supervisor = Supervisor()
timeStep = int(supervisor.getBasicTimeStep())

# Create the arm chain from the URDF
filename = None
with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as file:
    filename = file.name
    file.write(supervisor.getUrdf().encode('utf-8'))
armChain = Chain.from_urdf_file(filename, active_links_mask=[False, True, True, True, True, True, True, False])

# Initialize the arm motors and encoders
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

# Initialize fuzzy PID controllers for each active link
fuzzy_controllers = [JointFuzzyController(joint_name=f'joint_{i}') for i in range(len(armChain.active_links_mask))]

# Loop variables
elapsed_time = 0
loop_counter = 0
target_position = [1.0, 1.0, 1.0]  # Example target position
previous_errors = [0] * len(motors)
integral_of_error = [0] * len(motors)

print('Moving to target position...')

# Main loop
while supervisor.step(timeStep) != -1:
    elapsed_time += timeStep / 1000.0
    loop_counter += 1

    # Get joint angles for the defined target position
    joint_angles = get_joint_angles(target_position)

    # Command motors using fuzzy PID controller
    for i in range(min(len(motors), len(joint_angles))):
        current_angle = position_sensors[i].getValue()
        
        # Calculate error and change in error
        error = joint_angles[i] - current_angle
        error_change = error - previous_errors[i]
        
        # Compute velocity using fuzzy controller
        velocity = fuzzy_controllers[i].compute_velocity(error, error_change)

        # Update motor velocity and position
        motors[i].setVelocity(velocity)
        motors[i].setPosition(joint_angles[i])

        # Store previous error and update integral of error
        previous_errors[i] = error
        integral_of_error[i] += error * (timeStep / 1000.0)

    # Print joint angles every second
    if int(elapsed_time) % 1 == 0:
        recent_angles = get_recent_angles()
        print("Recent Joint Angles at t =", int(elapsed_time), "s:", recent_angles)

    # Break if maximum iterations are reached
    if loop_counter >= MAX_LOOP_ITERATIONS:
        print("Maximum loop iterations reached. Stopping.")
        break
