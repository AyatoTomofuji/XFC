import math
from typing import Tuple, Dict, Any, List
import random
from src.kessler_game.controller import KesslerController
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from func import aim, ast_angle, angle360
from src.kessler_game.ship import Ship


class NewController(KesslerController):
    """
    Class to be used by UC Fuzzy Challenge competitors to create a fuzzy logic controller
    for the Asteroid Smasher game.

    Note: Your fuzzy controller class can be called anything, but must inherit from
    the ``ControllerBase`` class (imported above)

    Users must define the following:
    1. __init__()
    2. actions(self, ship: SpaceShip, input_data: Dict[str, Tuple])

    By defining these interfaces, this class will work correctly
    """

    """
         A ship controller class for Kessler. This can be inherited to create custom controllers that can be passed to the
        game to operate within scenarios. A valid controller contains an actions method that takes in a ship object and ass
        game_state dictionary. This action method then sets the thrust, turn_rate, and fire commands on the ship object.
        """

    def __init__(self):
        """
        Create your fuzzy logic controllers and other objects here
        """

        #テストとしてTeam Asimovから引用
        A1 = ctrl.Antecedent(np.arange(0, 301, 1), 'A1')
        A1['near'] = fuzz.trimf(A1.universe, [-100, 0, 100])
        A1['close'] = fuzz.trimf(A1.universe, [0, 100, 200])
        A1['far'] = fuzz.trimf(A1.universe, [100, 200, 300])

        B1 = ctrl.Antecedent(np.arange(0, 361), 'B1')
        B1['front'] = fuzz.trimf(B1.universe, [-30, 0, 30])
        B1['middle'] = fuzz.trimf(B1.universe, [0,60, 120])
        B1['back'] = fuzz.trimf(B1.universe, [60, 210, 360])


        Con1 = ctrl.Consequent(np.arange(-26, 126, 1), 'Con1')
        Con1['Very Low'] = fuzz.trimf(Con1.universe, [-25, 0, 25])
        Con1['Low'] = fuzz.trimf(Con1.universe, [0, 25, 50])
        Con1['Medium'] = fuzz.trimf(Con1.universe, [25, 50, 75])
        Con1['High'] = fuzz.trimf(Con1.universe, [50, 75, 100])
        Con1['Very High'] = fuzz.trimf(Con1.universe, [75, 100, 125])

        Con1_rule1 = ctrl.Rule(antecedent=(A1['near'] & B1['front']),
                                    consequent=Con1['Very High'], label='Con1_rule1')
        Con1_rule2 = ctrl.Rule(antecedent=(A1['close'] & B1['front']),
                                    consequent=Con1['High'], label='Con1_rule2')
        Con1_rule3 = ctrl.Rule(antecedent=(A1['far'] & B1['front']),
                                    consequent=Con1['Medium'], label='Con1_rule3')
        Con1_rule4 = ctrl.Rule(antecedent=(A1['near'] & B1['middle']),
                                    consequent=Con1['Very High'], label='Con1_rule4')
        Con1_rule5 = ctrl.Rule(antecedent=(A1['close'] & B1['middle']),
                                    consequent=Con1['High'], label='Con1_rule5')
        Con1_rule6 = ctrl.Rule(antecedent=(A1['far'] & B1['middle']),
                                    consequent=Con1['Medium'], label='Con1_rule6')
        Con1_rule7 = ctrl.Rule(antecedent=(A1['near'] & B1['back']),
                                    consequent=Con1['Very High'], label='Con1_rule7')
        Con1_rule8 = ctrl.Rule(antecedent=(A1['close'] & B1['back']),
                                    consequent=Con1['High'], label='Con1_rule8')
        Con1_rule9 = ctrl.Rule(antecedent=(A1['far'] & B1['back']),
                                    consequent=Con1['Medium'], label='Con1_rule9')

        Con1_system = ctrl.ControlSystem(rules=[Con1_rule1, Con1_rule2, Con1_rule3, Con1_rule4,
                                                     Con1_rule5, Con1_rule6, Con1_rule7, Con1_rule8,
                                                     Con1_rule9])
        self.Con1_sim = ctrl.ControlSystemSimulation(Con1_system)

        A2 = ctrl.Antecedent(np.arange(0, 361), 'A2')
        A2['near'] = fuzz.trimf(A2.universe, [-15, 0, 15])
        A2['close'] = fuzz.trimf(A2.universe, [15, 45, 75])
        A2['far'] = fuzz.trimf(A2.universe, [75, 180, 285])
        # shortest_distance < 50 + (12 * clast_size)
        B2= ctrl.Antecedent(np.arange(0, 361), 'B2')
        B2['front'] = fuzz.trimf(B1.universe, [-30, 0, 30])
        B2['middle'] = fuzz.trimf(B1.universe, [0, 60, 120])
        B2['back'] = fuzz.trimf(B1.universe, [60, 210, 360])



        Con2 = ctrl.Consequent(np.arange(-26, 126, 1), 'Con2')
        Con2['Very Low'] = fuzz.trimf(Con2.universe, [-25, 0, 25])
        Con2['Low'] = fuzz.trimf(Con2.universe, [0, 25, 50])
        Con2['Medium'] = fuzz.trimf(Con2.universe, [25, 50, 75])
        Con2['High'] = fuzz.trimf(Con2.universe, [50, 75, 100])
        Con2['Very High'] = fuzz.trimf(Con2.universe, [75, 100, 125])
        Con2_rule1 = ctrl.Rule(antecedent=(A2['near'] & B2['front']),
                                    consequent=Con2['Very High'], label='Con2_rule1')
        Con2_rule2 = ctrl.Rule(antecedent=(A2['close'] & B2['front']),
                                    consequent=Con2['High'], label='Con2_rule2')
        Con2_rule3 = ctrl.Rule(antecedent=(A2['far'] & B2['front']),
                                    consequent=Con2['Medium'], label='Con2_rule3')
        Con2_rule4 = ctrl.Rule(antecedent=(A2['near'] & B2['middle']),
                                    consequent=Con2['Very High'], label='Con2_rule4')
        Con2_rule5 = ctrl.Rule(antecedent=(A2['close'] & B2['middle']),
                                    consequent=Con2['High'], label='Con2_rule5')
        Con2_rule6 = ctrl.Rule(antecedent=(A2['far'] & B2['middle']),
                                    consequent=Con2['Medium'], label='Con2_rule6')
        Con2_rule7 = ctrl.Rule(antecedent=(A2['near'] & B2['back']),
                                    consequent=Con2['Very High'], label='Con2_rule7')
        Con2_rule8 = ctrl.Rule(antecedent=(A2['close'] & B2['back']),
                                    consequent=Con2['High'], label='Con2_rule8')
        Con2_rule9 = ctrl.Rule(antecedent=(A2['far'] & B2['back']),
                                    consequent=Con2['Medium'], label='Con2_rule9')
        Con2_system = ctrl.ControlSystem(rules=[Con2_rule1, Con2_rule2, Con2_rule3, Con2_rule4,
                                                     Con2_rule5, Con2_rule6, Con2_rule7, Con2_rule8,
                                                     Con2_rule9])
        self.Con2_sim = ctrl.ControlSystemSimulation(Con2_system)



        B3 = ctrl.Antecedent(np.arange(0, 361, 1), 'B3')
        B3['front'] = fuzz.trimf(B3.universe, [-15, 0, 15])
        B3['middle'] = fuzz.trimf(B3.universe, [15, 45, 75])
        B3['back'] = fuzz.trimf(B3.universe, [75, 180, 285])
        A3 = ctrl.Antecedent(np.arange(0, 261, 1), 'A3')
        A3['near'] = fuzz.trimf(A3.universe, [-80, 0, 80])
        A3['close'] = fuzz.trimf(A3.universe, [80, 140, 200])
        A3['far'] = fuzz.trimf(A3.universe, [140, 200, 260])

        Con3 = ctrl.Consequent(np.arange(-26, 126, 1), 'Con3')
        Con3['Very Low'] = fuzz.trimf(Con3.universe, [-25, 0, 25])
        Con3['Low'] = fuzz.trimf(Con3.universe, [0, 25, 50])
        Con3['Medium'] = fuzz.trimf(Con3.universe, [25, 50, 75])
        Con3['High'] = fuzz.trimf(Con3.universe, [50, 75, 100])
        Con3['Very High'] = fuzz.trimf(Con3.universe, [75, 100, 125])

        Con3_rule1 = ctrl.Rule(antecedent=(A3['near'] & B3['front']),
                                    consequent=Con3['Very High'], label='Con3_rule1')

        Con3_rule2 = ctrl.Rule(antecedent=(A3['close'] & B3['front']),
                                    consequent=Con3['High'], label='Con3_rule2')

        Con3_rule3 = ctrl.Rule(antecedent=(A3['far'] & B3['front']),
                                    consequent=Con3['Medium'], label='Con3_rule3')

        Con3_rule4 = ctrl.Rule(antecedent=(A3['near'] & B3['middle']),
                                    consequent=Con3['Very High'], label='Con3_rule4')

        Con3_rule5 = ctrl.Rule(antecedent=(A3['close'] & B3['middle']),
                                    consequent=Con3['High'], label='Con3_rule5')

        Con3_rule6 = ctrl.Rule(antecedent=(A3['far'] & B3['middle']),
                                    consequent=Con3['Medium'], label='Con3_rule6')

        Con3_rule7 = ctrl.Rule(antecedent=(A3['near'] & B3['back']),
                                    consequent=Con3['Very High'], label='Con3_rule7')

        Con3_rule8 = ctrl.Rule(antecedent=(A3['close'] & B3['back']),
                                    consequent=Con3['High'], label='Con3_rule8')

        Con3_rule9 = ctrl.Rule(antecedent=(A3['far'] & B3['back']),
                                    consequent=Con3['Medium'], label='Con3_rule9')

        Con3_system = ctrl.ControlSystem(rules=[Con3_rule1, Con3_rule2, Con3_rule3, Con3_rule4,
                                                     Con3_rule5, Con3_rule6, Con3_rule7, Con3_rule8,
                                                     Con3_rule9])
        self.Con3_sim = ctrl.ControlSystemSimulation(Con3_system)

        B4 = ctrl.Antecedent(np.arange(0, 361), 'B4')
        B4['front'] = fuzz.trimf(B4.universe, [-15, 0, 15])
        B4['middle'] = fuzz.trimf(B4.universe, [15, 45, 75])
        B4['back'] = fuzz.trimf(B4.universe, [75, 180, 285])
        # shortest_distance < 50 + (12 * clast_size) We may wanna change the hypotenuse membership functions
        A4 = ctrl.Antecedent(np.arange(0, 361, 1), 'A4')

        A4['near'] = fuzz.trimf(A4.universe, [-80, 0, 80])
        A4['close'] = fuzz.trimf(A4.universe, [80, 140, 200])
        A4['far'] = fuzz.trimf(A4.universe, [140, 200, 260])

        Con4 = ctrl.Consequent(np.arange(-26, 126, 1), 'Con4')
        Con4['Very Low'] = fuzz.trimf(Con4.universe, [-25, 0, 25])
        Con4['Low'] = fuzz.trimf(Con4.universe, [0, 25, 50])
        Con4['Medium'] = fuzz.trimf(Con4.universe, [25, 50, 75])
        Con4['High'] = fuzz.trimf(Con4.universe, [50, 75, 100])
        Con4['Very High'] = fuzz.trimf(Con4.universe, [75, 100, 125])

        Con4_rule1 = ctrl.Rule(antecedent=(A4['near'] & B4['front']),
                                    consequent=Con4['Very High'], label='Con4_rule1')

        Con4_rule2 = ctrl.Rule(antecedent=(A4['close'] & B4['front']),
                                    consequent=Con4['High'], label='Con4_rule2')

        Con4_rule3 = ctrl.Rule(antecedent=(A4['far'] & B4['front']),
                                    consequent=Con4['Very Low'], label='Con4_rule3')

        Con4_rule4 = ctrl.Rule(antecedent=(A4['near'] & B4['middle']),
                                    consequent=Con4['High'], label='Con4_rule4')

        Con4_rule5 = ctrl.Rule(antecedent=(A4['close'] & B4['middle']),
                                    consequent=Con4['Medium'], label='Con4_rule5')

        Con4_rule6 = ctrl.Rule(antecedent=(A4['far'] & B4['middle']),
                                    consequent=Con4['Very Low'], label='Con4_rule6')

        Con4_rule7 = ctrl.Rule(antecedent=(A4['near'] & B4['back']),
                                    consequent=Con4['High'], label='Con4_rule7')

        Con4_rule8 = ctrl.Rule(antecedent=(A4['close'] & B4['back']),
                                    consequent=Con4['Low'], label='Con4_rule8')

        Con4_rule9 = ctrl.Rule(antecedent=(A4['far'] & B4['back']),
                                    consequent=Con4['Very Low'], label='Con4_rule9')

        Con4_system = ctrl.ControlSystem(rules=[Con4_rule1, Con4_rule2, Con4_rule3, Con4_rule4,
                                                     Con4_rule5, Con4_rule6, Con4_rule7, Con4_rule8,
                                                     Con4_rule9])
        self.Con4_sim = ctrl.ControlSystemSimulation(Con4_system)


        def r_angle(opposite, hypotenuse, abovebelow, leftright):
            if abovebelow > 0:
                angle = -1 * (math.degrees(math.asin(opposite / hypotenuse)))
            elif abovebelow < 0 and leftright < 0:
                angle = 180 + (math.degrees(math.asin(opposite / hypotenuse)))
            elif abovebelow < 0 and leftright > 0:
                angle = -180 + (math.degrees(math.asin(opposite / hypotenuse)))
            else:
                angle = 0
            return angle

        self.rangle = r_angle





    def actions(self, ownship: Ship, input_data: Dict[str, Tuple]) -> Tuple[float, float, bool]:
        # timeout(input_data)
        ship_list = input_data["ships"]
        ast_list = input_data["asteroids"]
        dist_list = [math.dist(ownship['position'], ast['position']) for ast in ast_list]
        closest = np.argmin(dist_list)
        ang = np.array(ast_list[closest]['position']) - np.array(ownship['position'])
        ang = angle360(math.degrees(np.arctan2(ang[1], ang[0])))
        print("あんぐ")
        print(ang)
        print(ownship['heading'])
        angdiff = ang - ownship['heading']
        if angdiff < -180: angdiff += 360
        elif angdiff > 180: angdiff -= 360
        print(angdiff)
        """for ship in ship_list:
        
            print(ship["id"], ship["team"], ship["position"])
            if ship["id"] != ownship.id:
                 print("Other ship at: ", ship["position"])
        """
        turn_rate = random.uniform(ownship['turn_rate_range'][0], ownship['turn_rate_range'][1])
        #もし最近接が前にいたら後退
        if abs(angdiff) < 30:
            thrust = random.uniform(ownship['thrust_range'][0], 0.0)
        elif abs(angdiff)>150:
            thrust = random.uniform(0.0, ownship['thrust_range'][1])
        else: thrust = random.uniform(ownship['thrust_range'][0], ownship['thrust_range'][1])
        #thrust_range = random.uniform(ownship['thrust_range'][0], ownship['thrust_range'][1])
        #fire_bullet = random.uniform(0.45, 1.0) < 0.5
        fire_bullet = abs(angdiff) < 15
        print(fire_bullet)
        speed = math.sqrt(ownship['velocity'][0]**2 + ownship['velocity'][1]**2)
        veloangle = math.degrees(math.atan2(ownship['velocity'][1], ownship['velocity'][0]))
        #print(ast_angle(self, ownship['position'], ast_list[closest]['position']))
        if speed > 0:
            if ownship['velocity'][1] > 0:
                travel_angle = -1 * math.degrees(math.asin(ownship['velocity'][0] / speed))
            elif ownship['velocity'][0] > 0:
                travel_angle = -180 + math.degrees(math.asin(ownship['velocity'][0] / speed))
            elif ownship['velocity'][0] < 0:
                travel_angle = 180 + math.degrees(math.asin(ownship['velocity'][0] / speed))
            else:
                travel_angle = 0



        sidefromcenter = 400 - ownship['position'][0]
        above_center = 300 - ownship['position'][1]
        distancefromcenter = ((ownship['position'][0] - 400) ** 2 + (ownship['position'][1] - 300) ** 2) ** 0.5
        if distancefromcenter > 0:
            anglefromcenter = self.rangle(sidefromcenter, distancefromcenter, above_center, sidefromcenter)
        else:
            anglefromcenter = 0

        ab2 = input_data['asteroids'][closest]['position'][1] - ownship['position'][1]
        lr2 = input_data['asteroids'][closest]['position'][0] - ownship['position'][0]
        op2 = lr2
        hyp2 = dist_list[closest]
        s_rangle_inrange = self.rangle(op2, hyp2, ab2, lr2)
        astangle = angle360(s_rangle_inrange)
        vx_mult = input_data['asteroids'][closest]['velocity'][0]
        vy_mult = input_data['asteroids'][closest]['velocity'][1]

        #前後，回転，射撃のタプルをリターンする
        return (thrust, turn_rate, fire_bullet)

    @property
    def name(self) -> str:
        return "OMU-CILab1"
