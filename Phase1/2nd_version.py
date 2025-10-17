import numpy as np
import sys
import tempfile
import math
from controller import Supervisor
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import ikpy
from ikpy.chain import Chain



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

