import sys
import tempfile
import math
from controller import Supervisor, DistanceSensor

try:
    import ikpy
    from ikpy.chain import Chain
except ImportError:
    sys.exit('The "ikpy" Python module is not installed. '
             'To run this sample, please upgrade "pip" and install ikpy with this command: "pip install ikpy"')

if ikpy.__version__[0] < '3':
    sys.exit('The "ikpy" Python module version is too old. '
             'Please upgrade "ikpy" Python module to version "3.0" or newer with this command: "pip install --upgrade ikpy"')

IKPY_MAX_ITERATIONS = 4

# Initialize the Webots Supervisor.
supervisor = Supervisor()
timeStep = int(4 * supervisor.getBasicTimeStep())

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

# Get the arm and target nodes.
target = supervisor.getFromDef('TARGET')
arm = supervisor.getSelf()

def get_joint_angles(target_position):
    """Calculate the joint angles to reach the desired target position."""
    initial_position = [0] + [m.getPositionSensor().getValue() for m in motors] + [0]
    joint_angles = armChain.inverse_kinematics(target_position, max_iter=IKPY_MAX_ITERATIONS, initial_position=initial_position)
    return joint_angles[1:]  # Exclude the base angle

def get_recent_angles():
    """Retrieve the recent angles of the arm joints from position sensors."""
    return [sensor.getValue() for sensor in position_sensors]

# Define the target position you want the robot to reach
target_position =  [0.5, 0.5, 0.2]  # Example target position (x, y, z)

# Loop variables
elapsed_time = 0
print('Moving to target position...')

# Main loop
while supervisor.step(timeStep) != -1:
    elapsed_time += timeStep / 1000.0  # Convert timeStep to seconds 
    
    # Call the function to get joint angles for the defined target position.
    joint_angles = get_joint_angles(target_position)
    
    # Debugging information
    print(f"Joint Angles: {joint_angles}")
    print(f"Number of Motors: {len(motors)}, Number of Joint Angles: {len(joint_angles)}")

    # Ensure we only set positions for available motors
    for i in range(min(len(motors), len(joint_angles))):
        motors[i].setPosition(joint_angles[i])

    # Print joint angles every second
    if int(elapsed_time) % 1 == 0:  # Check if 1 second has passed
        recent_angles = get_recent_angles()
        print("Calculated Joint Angles at t =", int(elapsed_time), "s:", joint_angles)
        print("Recent Joint Angles at t =", int(elapsed_time), "s:", recent_angles)
        
    # Optionally, add a condition to stop after reaching the target
    if all(abs(joint_angles[i] - recent_angles[i]) < 0.01 for i in range(min(len(joint_angles), len(recent_angles)))):
        print("Target position reached.")
        break
        