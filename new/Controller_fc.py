import math
from typing import Tuple, Dict

import joblib

from kessler_game.src.kesslergame import KesslerController, Ship
import numpy as np
from func import angle360, ast_angle

# XFC23用のコントローラー
# Fuzzy Controllerの入出力から再現したニューラルネットによるコントローラー


class Controller_neuro(KesslerController):
    """
         A ship controller class for Kessler. This can be inherited to create custom controllers that can be passed to the
        game to operate within scenarios. A valid controller contains an actions method that takes in a ship object and ass
        game_state dictionary. This action method then sets the thrust, turn_rate, and fire commands on the ship object.
        """


    def __init__(self):
        file = 'NNmodel.sav'

        self.loaded_model = joblib.load(file)
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

        if len(input_data['ships']) >= 2:
            angle_ships = ast_angle(ownship['position'], input_data['ships'][2 - ownship['id']]['position'])
            dist = math.dist(ownship['position'], input_data['ships'][2 - ownship['id']]['position'])
            if (dist <= avoidance):
                if angle_ships > 180:
                    angle_ships -= 360
                elif angle_ships < -180:
                    angle_ships += 360
                avoidance = dist
                angdiff = abs(angle_ships)


        # 前後，回転，射撃のタプルをリターンする(thrust…±480m/s^2 turn_rate…±180/s)
        result = self.loaded_model.predict([[avoidance, angdiff]])
        thrust = result[0][0]
        turn_rate = result[0][1]
        if turn_rate < -180:
            turn_rate = -180
        elif turn_rate > 180:
            turn_rate = 180

        if thrust > 480:
            thrust = 480
        elif thrust < -480:
            thrust = -480
        print(thrust)
        return thrust, turn_rate, fire_bullet, False

    @property
    def name(self) -> str:
        return "OMU-Let's"
