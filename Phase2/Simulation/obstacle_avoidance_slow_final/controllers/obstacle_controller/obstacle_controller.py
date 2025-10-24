from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Get the distance sensor
sensor = robot.getDevice('ObstacleSensor')
sensor.enable(timestep)

while robot.step(timestep) != -1:
    distance = sensor.getValue()
    print(f"Obstacle detected object at distance: {distance}")
