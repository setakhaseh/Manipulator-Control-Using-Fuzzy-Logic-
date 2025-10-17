# Robotic Manipulator Control Using Fuzzy Logic

## Overview
This project implements a fuzzy logic controller for a robotic manipulator in simulation using **Webots**. It was developed for the **Computational Intelligence** course at Amirkabir University of Technology, Faculty of Electrical Engineering.

The manipulator is controlled autonomously to perform tasks such as reaching a target, avoiding obstacles, and respecting joint constraints. Fuzzy logic rules allow adaptive control based on sensor inputs.

## Features
- Phase 1: Control arm to a fixed target using simple fuzzy rules.
- Phase 2: Handle dynamic targets and fixed obstacles with obstacle avoidance.
- Phase 3: Advanced fuzzy control with moving obstacles, dynamic target tracking, and joint constraints.
- Input from simulated sensors: joint angles, velocities, distances to target and obstacles.
- Output: adaptive joint movement commands.

## Simulation Environment
- **Webots** is used for realistic multi-body dynamics simulation.
- The robot arm interacts with objects and obstacles in a configurable workspace.
- Sensor data from Webots (e.g., joint angles, velocities, distances) are fed to the fuzzy controller.

## File Structure
- `controller_basic.py` – Phase 1 fuzzy controller.
- `controller_additional.py` – Phase 2 fuzzy controller for obstacles.
- `controller_advanced.py` – Phase 3 advanced fuzzy controller.
- `rule_additional.txt` – Fuzzy rules for Phase 2.
- `rule_advanced.txt` – Fuzzy rules for Phase 3.
- `notebooks/` – Optional Jupyter notebooks for testing and visualization.
- `data/` – Any sample data or scenarios used for testing.

## How to Run
1. Install Webots and required Python packages.
2. Load the robot manipulator environment in Webots.
3. Run the Python controller file corresponding to the desired phase.
4. Observe the manipulator behavior in the Webots simulation window.

## Notes
- Users can modify membership functions and fuzzy rules to test different behaviors.
- The simulation allows testing in static and dynamic environments.
- Designed for academic purposes in the Computational Intelligence course.

