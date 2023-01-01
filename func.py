#関数置き場
import math
from typing import Tuple, Dict, Any, List
import random
from src.kessler_game.controller import KesslerController
from src.kessler_game.ship import Ship
import numpy as np

def aim(vx, vy, normal_astangle):
    #asteroid velocity
    veloang = math.degrees(math.atan2(vy,vx))
    velocity = math.sqrt(vx**2 + vy**2)
    bullet_speed = 800

    # Find angle B
    tsangle = normal_astangle - 90  # This is the ship relative to the asteroid, don't change this if you're looking at it backwards.
    if tsangle <= 0:
        tsangle += 360

    orientation_ast = (tsangle - veloang)  # Vectang: asteroid travel angle, Trangle: True asteroid angle to ship

    if orientation_ast >= 180:
        orientation_ast = -360 + orientation_ast
    elif orientation_ast <= -180:
        orientation_ast = 360 + orientation_ast

    ant_angle = -70 * math.degrees(
        math.asin(math.sin(math.radians(orientation_ast)) * sidea / sideb))  # Calculates anticipatory aiming.

    """This next line of code can be used to disable anticipatory aiming"""
    # self.ant_angle = 0

    return ant_angle


def asteroid_dist(self, ownship: Ship, input_data: Dict[str, Tuple]) -> float:
    ship_list = input_data["ships"]
    ast_list = input_data["asteroids"]
    self.position = ship_list[ownship.team - 1]['position']
    dist_list = []
    for ship in ship_list:
        for ast in ast_list:
            dist_list.append(math.dist(ship['position'], ast['position']))
            print(ast['velocity'])
    print(dist_list)
    closest = np.argmin(dist_list)

def ast_angle(self, ownship_position, ast_position):
    ang = np.array(ast_position) - np.array(ownship_position)
    ang = np.arctan2(ang[1], ang[0])
    return math.degrees(ang)

def anglechange(ast_angle, ship_angle):
    diff_angle = ast_angle - ship_angle
    if diff_angle >= 180:
        ans_angle = 360 - diff_angle
        flag = 1 #1は時計回り
    elif diff_angle>=0:
        ans_angle = diff_angle
        flag = 0 #0は正の回転
    elif diff_angle < -180:
        ans_angle = 360 + diff_angle
        flag = 0
    else:
        ans_angle = -diff_angle
        flag = 1
    return ans_angle, flag #

def anglechange2(ast_angle, ship_angle):
    diff_angle = ast_angle - ship_angle
    if diff_angle > 180:
        ans_angle = 360 - diff_angle
    elif diff_angle < -180:
        ans_angle = -diff_angle - 360
    return ans_angle #角度がプラスなら左回り，マイナスなら右回り　上の関数の角度と左右をまとめたもの

def angle360(angle):
    if angle < 0:
        angle += 360
    return angle


