"""obstacle_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get the distance sensor
sensor = robot.getDevice('ObstacleSensor')
sensor.enable(timestep)

while robot.step(timestep) != -1:
    distance = sensor.getValue()
    print(f"Obstacle detected object at distance: {distance}")