import math
from typing import Tuple, Dict
from kessler_game.src.kesslergame import KesslerController, Ship
import numpy as np
from func import angle360, ast_angle
from sklearn.neural_network import MLPRegressor


# 距離5分割，センターアングルは2分割，隕石の速度とサイズを入力，MFの遺伝子のみを正規化
class input4Controller(KesslerController):
    """
         A ship controller class for Kessler. This can be inherited to create custom controllers that can be passed to the
        game to operate within scenarios. A valid controller contains an actions method that takes in a ship object and ass
        game_state dictionary. This action method then sets the thrust, turn_rate, and fire commands on the ship object.
        """

    def __init__(self, gene, genes2):
        """
        Create your fuzzy logic controllers and other objects here
        """
        # gene: [left, center, right, (angle)center]
        left1 = gene[0]*1000
        left2 = gene[1]*1000
        center = gene[2]*1000
        right2 = gene[3]*1000
        right1 = gene[4]*1000
        speed1 = gene[5]*200
        speed2 = gene[6]*200
        speed3 = gene[7]*200
        speed4 = gene[8]*200
        speed5 = gene[9]*200
        size = gene[10]
        center_angle = gene[11]*180



        def membership1(x):
            if x <= left1:
                return [1.0, 0.0, 0.0, 0.0, 0.0]
            elif x <= left2:
                return np.array([1.0 - (x - left1) / (left2 - left1), (x - left1) / (left2 - left1), 0.0, 0.0, 0.0])
            elif x <= center:
                return np.array([0.0, (center - x) / (center - left2),
                                 (x - left2) / (center - left2), 0.0, 0.0])
            elif x <= right2:
                return np.array([0.0, 0.0, (right2 - x) / (right2 - center), (x - center) / (right2 - center), 0.0])
            elif x <= right1:
                return np.array([0.0, 0.0, 0.0, (right1 - x) / (right1 - right2), (x - right2) / (right1 - right2)])
            else:
                return np.array([0.0, 0.0, 0.0, 0.0, 1.0])

        def membership2(angle):
            angle = abs(angle)
            if angle <= center_angle:
                return np.array([1.0 - angle / center_angle, angle / center_angle, 0.0])
            elif angle <= 180:
                return np.array([0.0, 2 - angle / center_angle, angle / center_angle - 1])



        self.membership1 = membership1
        self.membership2 = membership2

        def mems(x, angle, speed):
            nonlocal genes2
            genes_out = np.array(genes2).copy()
            # ルールは75個なので，outputの数値は150個になる
            outputs = [[genes_out[i], genes_out[i+75]] for i in range(75)]

            # distance
            k = membership1(x)
            # angle
            p = membership2(angle)
            # speed
            l = membership1(speed)
            #サイズは5*3*5=75

            rules = np.outer(np.outer(k, p).flatten(), l).flatten()
            return sum(outputs * rules)

        self.mems = mems

        center_x = 500
        center_y = 400

    def actions(self, ownship: Dict, input_data: Dict[str, Tuple]) -> Tuple[float, float, bool]:
        # timeout(input_data)
        # 隕石と機体の位置関係のセクション
        ast_list = np.array(input_data["asteroids"])
        # (x,y)で表す，機体からの距離
        dist_xylist = [np.array(ownship['position']) - np.array(ast['position']) for ast in ast_list]
        dist_avoid_list = dist_xylist.copy()
        dist_list1 = [math.sqrt(xy[0] ** 2 + xy[1] ** 2) for xy in dist_xylist]

        # よける部分に関しては画面端のことを考える，弾丸はすり抜けないから狙撃に関しては考えない
        sidefromcenter = 500 - ownship['position'][0]
        below_center = 400 - ownship['position'][1]
        for xy in dist_avoid_list:
            if xy[0] > 500:
                xy[0] -= 1000
            elif xy[0] < -500:
                xy[0] += 1000
            if xy[1] > 400:
                xy[1] -= 800
            elif xy[1] < -400:
                xy[1] += 800
        dist_avoid_list = [math.sqrt(xy[0] ** 2 + xy[1] ** 2) for xy in dist_avoid_list]

        sorted2_idx = np.argsort(dist_avoid_list)
        sorteddict = ast_list[sorted2_idx]
        # ここから考えるのは近傍5つの隕石
        search_list = sorteddict[0:5]
        search_dist = np.array([math.dist(ownship['position'], ast['position']) for ast in search_list])
        angle_dist = [np.array(ast['position']) - np.array(ownship['position']) for ast in search_list]
        angle_dist = [angle360(math.degrees((np.arctan2(near_ang[1], near_ang[0])))) - ownship['heading'] for near_ang
                      in angle_dist]
        aalist = []
        for ang in angle_dist:
            if ang > 180:
                ang -= 360
            elif ang < -180:
                ang += 360
            aalist.append(ang)

        angdiff_front = min(aalist, key=abs)
        angdiff = aalist[0]
        fire_bullet = abs(angdiff_front) < 10 and min(dist_list1) < 400
        avoidance = np.min(dist_avoid_list)
        speed = math.sqrt(ownship['velocity'][0] ** 2 + ownship['velocity'][1] ** 2)
        if len(input_data['ships']) >= 2:
            angle_ships = ast_angle(ownship['position'], input_data['ships'][2 - ownship['id']]['position'])
            dist = math.dist(ownship['position'], input_data['ships'][2 - ownship['id']]['position'])
            speed = math.sqrt(input_data['ships'][2 - ownship['id']]['velocity'][0] ** 2 + input_data['ships'][2 - ownship['id']]['velocity'][1] ** 2)
            if (dist <= avoidance):
                if angle_ships > 180:
                    angle_ships -= 360
                elif angle_ships < -180:
                    angle_ships += 360
                avoidance = dist
                angdiff = abs(angle_ships)

        model_thrust = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000)
        model_angular_velocity = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=1000)

        rule = self.mems(avoidance, angdiff, speed)

        thrust = rule[0]
        turn_rate = rule[1] * np.sign(angdiff)

        if thrust > ownship['thrust_range'][1]:
            thrust = ownship['thrust_range'][1]
        elif thrust < ownship['thrust_range'][0]:
            thrust = ownship['thrust_range'][0]
        if turn_rate > ownship['turn_rate_range'][1]:
            turn_rate = ownship['turn_rate_range'][1]
        elif turn_rate < ownship['turn_rate_range'][0]:
            turn_rate = ownship['turn_rate_range'][0]
        # 前後，回転，射撃のタプルをリターンする(thrust…±480m/s^2 turn_rate…±180/s)
        return thrust, turn_rate, fire_bullet



    @property
    def name(self) -> str:
        return "OMU-Let's"
