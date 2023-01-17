import math
from typing import Tuple, Dict
import random
from kesslergame import KesslerController
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from func import angle360
from kesslergame import Ship


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

    def __init__(self, gene):
        """
        Create your fuzzy logic controllers and other objects here
        """
        #gene: [left, center, right, (angle)center]
        left = gene[0]
        center = gene[1]
        right = gene[2]
        center_angle = gene[3]
        def membership1(x):
            if x <= gene[1]:
                return np.array([1.0-(x-left)/(center - left), (x-left)/(center - left), 0.0])
            elif x<= 2 * center:
                return np.array([0.0, right/(right-center)-x/(right-center), x/(right-center)-center/(right-center)])
            else:
                return np.array([0.0, 0.0, 1.0])

        def membership2(angle):
            angle = abs(angle)
            if angle <= 90:
                return np.array([1.0-angle/center_angle, angle/center_angle, 0.0])
            elif angle <= 180:
                return np.array([0.0, 2-angle/center_angle, angle/center_angle-1])

        self.membership1 = membership1
        self.membership2 = membership2
        self.center = gene[1]

        def mems(x, angle, gene):
            Rules = {
            0: "x1 is near  and x2 is front ",
            1: "x1 is near  and x2 is middle",
            2: "x1 is near  and x2 is back" ,
            3: "x1 is close and x2 is front",
            4: "x1 is close and x2 is middle",
            5: "x1 is close and x2 is back",
            6: "x1 is far   and x2 is front",
            7: "x1 is far   and x2 is middle",
            8: "x1 is far   and x2 is back",
            }
            out0 = np.array([-480, 90])
            out1 = np.array([-360, 180])
            out2 = np.array([120, 180])
            out3 = np.array([-360, 180])
            out4 = np.array([-60, 180])
            out5 = np.array([180, 180])
            out6 = np.array([120, 180])
            out7 = np.array([120, 180])
            out8 = np.array([0, 180])

            k = membership1(x)
            p = membership2(angle)

            rule0 = k[0] * p[0]
            rule1 = k[0] * p[1]
            rule2 = k[0] * p[2]
            rule3 = k[1] * p[0]
            rule4 = k[1] * p[1]
            rule5 = k[1] * p[2]
            rule6 = k[2] * p[0]
            rule7 = k[2] * p[1]
            rule8 = k[2] * p[2]
            out = ((rule0 * out0) + (rule1 * out1) + (rule2 * out2) + (rule3 * out3) + (rule4 * out4) + (rule5 * out5) + (rule6 * out6) + (
                        rule7 * out7) + (rule8 * out8))
            return out
        self.mems = mems



    def actions(self, ownship: Ship, input_data: Dict[str, Tuple]) -> Tuple[float, float, bool]:
        # timeout(input_data)
        #ship_list = input_data["ships"]
        ast_list = input_data["asteroids"]
        dist_list = np.array([math.dist(ownship['position'], ast['position']) for ast in ast_list])
        closest = np.argmin(dist_list)
        dist_closest = dist_list[closest]







        #fire_bullet = abs(angdiff) < 10.0


        # print(ast_angle(self, ownship['position'], ast_list[closest]['position']))



        #search_list = [ast_list[num] for num in np.where(np.array(dist_list) < search_range)[0]]
        #search_dist = [dist for dist in dist_list if dist < search_range]
        sorteddict = sorted(ast_list, key=lambda x:math.dist(ownship['position'], x['position']))
        search_list = sorteddict[0:5]
        search_dist = np.array([math.dist(ownship['position'], ast['position']) for ast in search_list])
        near_angle = [np.array(ast['position']) - np.array(ownship['position']) for ast in search_list]
        near_angle = [angle360(math.degrees((np.arctan2(near_ang[1], near_ang[0])))) - ownship['heading'] for near_ang in near_angle]
        aalist = []
        for ang in near_angle:
            if ang > 180:
                ang -= 360
            elif ang < -180:
                ang += 360
            aalist.append(ang)

        angdiff_front = min(aalist, key=abs)
        angdiff = aalist[np.argmin(search_dist)]
        fire_bullet = abs(angdiff_front) < 10 and min(dist_list) < 300
        ruru = self.mems(dist_closest,angdiff, self.center)
        thrust = ruru[0]
        turn_rate = ruru[1]*np.sign(angdiff)

        # Team1がよける
        if len(input_data['ships']) >= 2 and ownship['id'] == 1:
            if math.dist(ownship['position'], input_data['ships'][1]['position']) < 100:
                thrust *= -1



        # 前後，回転，射撃のタプルをリターンする(thrust…±480m/s^2 turn_rate…±180/s)
        return (thrust, turn_rate, fire_bullet)

    @property
    def name(self) -> str:
        return "Optimized Genes"
