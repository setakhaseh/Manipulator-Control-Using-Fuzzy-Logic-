import numpy as np
import sys
import tempfile
import math
from controller import Supervisor
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import ikpy
from ikpy.chain import Chain

# Define the JointFuzzyController class
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
        self.e['NM'] = fuzz.trimf(self.e.universe, [-1.5, -0.75, 0])
        self.e['NS'] = fuzz.trimf(self.e.universe, [-0.75, 0, 0.75])
        self.e['ZO'] = fuzz.trimf(self.e.universe, [-1.5, 0, 1.5])
        self.e['PS'] = fuzz.trimf(self.e.universe, [0, 0.75, 1.5])
        self.e['PM'] = fuzz.trimf(self.e.universe, [0.75, 1.5, 1.5])
        self.e['PB'] = fuzz.trimf(self.e.universe, [0.75, 1.5, 1.5])
        
        # Membership functions for error change (ec)
        self.ec['NB'] = fuzz.trimf(self.ec.universe, [-1.5, -1.5, -0.75])
        self.ec['NM'] = fuzz.trimf(self.ec.universe, [-1.5, -0.75, 0])
        self.ec['NS'] = fuzz.trimf(self.ec.universe, [-0.75, 0, 0.75])
        self.ec['ZO'] = fuzz.trimf(self.ec.universe, [-1.5, 0, 1.5])
        self.ec['PS'] = fuzz.trimf(self.ec.universe, [0, 0.75, 1.5])
        self.ec['PM'] = fuzz.trimf(self.ec.universe, [0.75, 1.5, 1.5])
        self.ec['PB'] = fuzz.trimf(self.ec.universe, [0.75, 1.5, 1.5])

        # Membership functions for velocity
        self.velocity['NB'] = fuzz.trimf(self.velocity.universe, [-1.5, -1.5, -0.75])
        self.velocity['NM'] = fuzz.trimf(self.velocity.universe, [-1.5, -0.75, 0])
        self.velocity['NS'] = fuzz.trimf(self.velocity.universe, [-0.75, 0, 0.75])
        self.velocity['ZO'] = fuzz.trimf(self.velocity.universe, [-1.5, 0, 1.5])
        self.velocity['PS'] = fuzz.trimf(self.velocity.universe, [0, 0.75, 1.5])
        self.velocity['PM'] = fuzz.trimf(self.velocity.universe, [0.75, 1.5, 1.5])
        self.velocity['PB'] = fuzz.trimf(self.velocity.universe, [0.75, 1.5, 1.5])

    def _init_rules(self):
        # Define rules for velocity control
        self.rules = [
            ctrl.Rule(self.e['PB'] & self.ec['NB'], self.velocity['ZO']),
            ctrl.Rule(self.e['PB'] & self.ec['NM'], self.velocity['ZO']),
            ctrl.Rule(self.e['PB'] & self.ec['NS'], self.velocity['NS']),
            ctrl.Rule(self.e['PB'] & self.ec['ZO'], self.velocity['NM']),
            ctrl.Rule(self.e['PB'] & self.ec['PS'], self.velocity['NM']),
            ctrl.Rule(self.e['PB'] & self.ec['PM'], self.velocity['NB']),
            ctrl.Rule(self.e['PB'] & self.ec['PB'], self.velocity['NB']),
            ctrl.Rule(self.e['PM'] & self.ec['NB'], self.velocity['PS']),
            ctrl.Rule(self.e['PM'] & self.ec['NM'], self.velocity['ZO']),
            ctrl.Rule(self.e['PM'] & self.ec['NS'], self.velocity['NS']),
            ctrl.Rule(self.e['PM'] & self.ec['ZO'], self.velocity['NM']),
            ctrl.Rule(self.e['PM'] & self.ec['PS'], self.velocity['NM']),
            ctrl.Rule(self.e['PM'] & self.ec['PM'], self.velocity['NM']),
            ctrl.Rule(self.e['PM'] & self.ec['PB'], self.velocity['NB']),
            ctrl.Rule(self.e['PS'] & self.ec['NB'], self.velocity['PS']),
            ctrl.Rule(self.e['PS'] & self.ec['NM'], self.velocity['PS']),
            ctrl.Rule(self.e['PS'] & self.ec['NS'], self.velocity['ZO']),
            ctrl.Rule(self.e['PS'] & self.ec['ZO'], self.velocity['NS']),
            ctrl.Rule(self.e['PS'] & self.ec['PS'], self.velocity['NM']),
            ctrl.Rule(self.e['PS'] & self.ec['PM'], self.velocity['NM']),
            ctrl.Rule(self.e['PS'] & self.ec['PB'], self.velocity['NM']),
            ctrl.Rule(self.e['ZO'] & self.ec['NB'], self.velocity['PM']),
            ctrl.Rule(self.e['ZO'] & self.ec['NM'], self.velocity['PM']),
            ctrl.Rule(self.e['ZO'] & self.ec['NS'], self.velocity['PS']),
            ctrl.Rule(self.e['ZO'] & self.ec['ZO'], self.velocity['ZO']),
            ctrl.Rule(self.e['ZO'] & self.ec['PS'], self.velocity['NS']),
            ctrl.Rule(self.e['ZO'] & self.ec['PM'], self.velocity['NS']),
            ctrl.Rule(self.e['ZO'] & self.ec['PB'], self.velocity['NM']),
            ctrl.Rule(self.e['NS'] & self.ec['NB'], self.velocity['PB']),
            ctrl.Rule(self.e['NS'] & self.ec['NM'], self.velocity['PM']),
            ctrl.Rule(self.e['NS'] & self.ec['NS'], self.velocity['PM']),
            ctrl.Rule(self.e['NS'] & self.ec['ZO'], self.velocity['PS']),
            ctrl.Rule(self.e['NS'] & self.ec['PS'], self.velocity['ZO']),
            ctrl.Rule(self.e['NS'] & self.ec['PM'], self.velocity['NS']),
            ctrl.Rule(self.e['NS'] & self.ec['PB'], self.velocity['NS']),
            ctrl.Rule(self.e['NM'] & self.ec['NB'], self.velocity['PB']),
            ctrl.Rule(self.e['NM'] & self.ec['NM'], self.velocity['PM']),
            ctrl.Rule(self.e['NM'] & self.ec['NS'], self.velocity['PM']),
            ctrl.Rule(self.e['NM'] & self.ec['ZO'], self.velocity['PS']),
            ctrl.Rule(self.e['NM'] & self.ec['PS'], self.velocity['PS']),
            ctrl.Rule(self.e['NM'] & self.ec['PM'], self.velocity['ZO']),
            ctrl.Rule(self.e['NM'] & self.ec['PB'], self.velocity['NS']),
            ctrl.Rule(self.e['NB'] & self.ec['NB'], self.velocity['PB']),
            ctrl.Rule(self.e['NB'] & self.ec['NM'], self.velocity['PB']),
            ctrl.Rule(self.e['NB'] & self.ec['NS'], self.velocity['PM']),
            ctrl.Rule(self.e['NB'] & self.ec['ZO'], self.velocity['PM']),
            ctrl.Rule(self.e['NB'] & self.ec['PS'], self.velocity['PS']),
            ctrl.Rule(self.e['NB'] & self.ec['PM'], self.velocity['ZO']),
            ctrl.Rule(self.e['NB'] & self.ec['PB'], self.velocity['ZO'])
        ]

    def compute_velocity(self, error, error_change):
        self.sim.input[f'{self.e.label}'] = error
        self.sim.input[f'{self.ec.label}'] = error_change
        self.sim.compute()
        return self.sim.output[f'{self.velocity.label}']

