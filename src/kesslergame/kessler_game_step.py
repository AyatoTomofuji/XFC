# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time

import gym
import numpy as np
from typing import Dict, Any, List
from enum import Enum
from collections import OrderedDict

import torch

from .score import Score
from .controller import KesslerController
from .collisions import circle_line_collision
from .graphics import GraphicsType, GraphicsHandler


class StopReason(Enum):
    not_stopped = 0
    no_ships = 1
    no_asteroids = 2
    time_expired = 3
    out_of_bullets = 4


class Game_step:
    def __init__(self, settings: Dict[str, Any] = None,  ):

        if settings is None:
            settings = {
            'graphics_type': GraphicsType.NoGraphics,

            }

        self.action_space = gym.spaces.Box(low=np.array([-480, -180]), high=np.array([480, 180]), shape=(2,), dtype=np.float32)
        # Game settings
        self.frequency = settings.get("frequency", 30.0)
        self.time_step = 1 / settings.get("frequency", 30.0)
        self.perf_tracker = settings.get("perf_tracoidker", True)
        self.prints_on = settings.get("prints_on", True)
        self.graphics_type = settings.get("graphics_type", GraphicsType.Tkinter)
        self.graphics_obj = settings.get("graphics_obj", None)
        self.realtime_multiplier = settings.get("realtime_multiplier", 0 if self.graphics_type==GraphicsType.NoGraphics else 1)
        self.time_limit = settings.get("time_limit", None)

        # UI settings
        default_ui = {'ships': True, 'lives_remaining': True, 'accuracy': True,
                      'asteroids_hit': True, 'bullets_remaining': True, 'controller_name': True}
        self.UI_settings = settings.get("UI_settings", default_ui)
        if self.UI_settings == 'all':
            self.UI_settings = {'ships': True, 'lives_remaining': True, 'accuracy': True,
                                'asteroids_hit': True, 'shots_fired': True, 'bullets_remaining': True,
                                'controller_name': True}




    def reset(self, scenario=None, controllers=None):
        self.scenario = scenario
        self.asteroids = scenario.asteroids()
        self.ships = scenario.ships()
        self.bullets = []
        self.mines = []


        # Initialize Scoring class
        self.score = Score(scenario)
        self.controllers = controllers

        # Initialize environment parameters
        self.stop_reason = StopReason.not_stopped
        self.sim_time = 0
        self.step = 0
        self.time_limit = scenario.time_limit if scenario.time_limit else self.time_limit
        for controller, ship in zip(self.controllers, self.ships):
            controller.ship_id = ship.id
            ship.controller = controller
        liveships = [ship for ship in self.ships if ship.alive]

        game_state = {
            'asteroids': [asteroid.state for asteroid in self.asteroids],
            'ships': [ship.state for ship in liveships],
            'bullets': [bullet.state for bullet in self.bullets],
            'mines': [mine.state for mine in self.mines],
            'map_size': self.scenario.map_size,
            'time': self.sim_time,
            'time_step': self.step,
        }
        state = controllers[0].actions(self, ownship=self.ships[0].ownstate, input_data=game_state)
        state = torch.tensor(state, dtype=torch.float32)
        return state
    def run_step(self, actions):



        # Run an entire scenario from start to finish and return score and stop reason

        ##################
        # INITIALIZATION #
        ##################
        # Initialize objects lists from scenario

        # Initialize performance tracker
        # Assign controllers to each ship
        for controller, ship in zip(self.controllers, self.ships):
            controller.ship_id = ship.id
            ship.controller = controller

        # Initialize graphics display
        graphics = GraphicsHandler(type=self.graphics_type, scenario=self.scenario, UI_settings=self.UI_settings, graphics_obj=self.graphics_obj)

        # Initialize list of dictionary for performance tracking (will remain empty if perf_tracker is false
        perf_list = []
        # Get perf time at the start of time step evaluation and initialize performance tracker
        step_start = time.perf_counter()
        perf_dict = OrderedDict()

        # --- CALL CONTROLLER FOR EACH SHIP ------------------------------------------------------------------------
        # Get all live ships
        liveships = [ship for ship in self.ships if ship.alive]

        game_state = {
            'asteroids': [asteroid.state for asteroid in self.asteroids],
            'ships': [ship.state for ship in liveships],
            'bullets': [bullet.state for bullet in self.bullets],
            'mines': [mine.state for mine in self.mines],
            'map_size': self.scenario.map_size,
            'time': self.sim_time,
            'time_step': self.step,
        }

        ######################
        # MAIN SCENARIO LOOP #
        ######################



        # Generate game_state info to send to controllers
        # Initialize controller time recording in performance tracker
        if self.perf_tracker:
            perf_dict['controller_times'] = []
            t_start = time.perf_counter()
        # Loop through each controller/ship combo and apply their actions
        for idx, ship in enumerate(self.ships):
            if ship.alive:
                # Reset controls on ship to defaults
                ship.thrust = 0
                ship.turn_rate = 0
                ship.fire = False
                # Evaluate each controller letting control be applied
                if self.controllers[idx].ship_id != ship.id:
                    raise RuntimeError("Controller and ship ID do not match")
                states = self.controllers[idx].actions(self, ship.ownstate, game_state)
                ship.thrust = actions[0][0]
                ship.turn_rate = actions[0][1]
                ship.fire = True
                ship.mine = False
            # Update controller evaluation time if performance tracking
            if self.perf_tracker:
                controller_time = time.perf_counter() - t_start if ship.alive else 0.00
                perf_dict['controller_times'].append(controller_time)
                t_start = time.perf_counter()
        if self.perf_tracker:
            perf_dict['total_controller_time'] = time.perf_counter() - step_start
            prev = time.perf_counter()
        # --- UPDATE STATE INFORMATION OF EACH OBJECT --------------------------------------------------------------
        # Update each Asteroid, Bullet, and Ship
        for bullet in self.bullets:
            bullet.update(self.time_step)
        for mine in self.mines:
            mine.update(self.time_step)
        for asteroid in self.asteroids:
            asteroid.update(self.time_step)
        for ship in liveships:
            if ship.alive:
                new_bullet, new_mine = ship.update(self.time_step)
                if new_bullet is not None:
                    self.bullets.append(new_bullet)
                if new_mine is not None:
                    self.mines.append(new_mine)
        # Cull any bullets past the map edge
        bullets = [bullet
                   for bullet
                   in self.bullets
                   if 0 <= bullet.position[0] <= self.scenario.map_size[0]
                   and 0 <= bullet.position[1] <= self.scenario.map_size[1]]
        # Wrap ships and asteroids to other side of map
        for ship in liveships:
            for idx, pos in enumerate(ship.position):
                bound = self.scenario.map_size[idx]
                offset = bound - pos
                if offset < 0 or offset > bound:
                    ship.position[idx] += bound * np.sign(offset)
        for asteroid in self.asteroids:
            for idx, pos in enumerate(asteroid.position):
                bound = self.scenario.map_size[idx]
                offset = bound - pos
                if offset < 0 or offset > bound:
                    asteroid.position[idx] += bound * np.sign(offset)
        # Update performance tracker with
        if self.perf_tracker:
            perf_dict['physics_update'] = time.perf_counter() - prev
            prev = time.perf_counter()
        # --- CHECK FOR COLLISIONS ---------------------------------------------------------------------------------
        # --- Check asteroid-bullet collisions ---
        bullet_remove_idxs = []
        asteroid_remove_idxs = []
        check = 0
        for idx_bul, bullet in enumerate(bullets):
            for idx_ast, asteroid in enumerate(self.asteroids):
                # If collision occurs
                if circle_line_collision(bullet.position, bullet.tail, asteroid.position, asteroid.radius):
                    # Increment hit values on ship that fired bullet then destruct bullet and mark for removal
                    bullet.owner.asteroids_hit += 1
                    bullet.owner.bullets_hit += 1
                    bullet.destruct()
                    print("hit")
                    check = 1
                    bullet_remove_idxs.append(idx_bul)
                    # Asteroid destruct function and mark for removal
                    self.asteroids.extend(asteroid.destruct(impactor=bullet))
                    asteroid_remove_idxs.append(idx_ast)
                    # Stop checking this bullet
                    break
        # Cull bullets and asteroids that are marked for removal
        bullets = [bullet for idx, bullet in enumerate(bullets) if idx not in bullet_remove_idxs]
        asteroids = [asteroid for idx, asteroid in enumerate(self.asteroids) if idx not in asteroid_remove_idxs]
        # --- Check mine-asteroid and mine-ship effects ---
        mine_remove_idxs = []
        asteroid_remove_idxs = []
        new_asteroids = []
        for idx_mine, mine in enumerate(self.mines):
            if mine.detonating:
                for idx_ast, asteroid in enumerate(asteroids):
                    dist = np.sqrt((asteroid.position[0] - mine.position[0]) ** 2 + (asteroid.position[1] - mine.position[1]) ** 2)
                    if dist <= mine.blast_radius + asteroid.radius:
                        mine.owner.asteroids_hit += 1
                        mine.owner.mines_hit += 1
                        new_asteroids.extend(asteroid.destruct(impactor=mine))
                        asteroid_remove_idxs.append(idx_ast)
                for ship in liveships:
                    dist = np.sqrt((ship.position[0] - mine.position[0]) ** 2 + (ship.position[1] - mine.position[1]) ** 2)
                    if dist <= mine.blast_radius + ship.radius:
                        # Ship destruct function. Add one to asteroids_hit
                        ship.destruct(map_size=self.scenario.map_size)
                        # Stop checking this ship's collisions
                        break
                if idx_mine not in mine_remove_idxs:
                    mine_remove_idxs.append(idx_mine)
                mine.destruct()
        mines = [mine for idx, mine in enumerate(self.mines) if idx not in mine_remove_idxs]
        asteroids = [asteroid for idx, asteroid in enumerate(asteroids) if idx not in asteroid_remove_idxs]
        asteroids.extend(new_asteroids)
        # --- Check asteroid-ship collisions ---
        asteroid_remove_idxs = []
        for idx_ship, ship in enumerate(liveships):
            if not ship.is_respawning:
                for idx_ast, asteroid in enumerate(asteroids):
                    dist = np.sqrt(sum([(pos1 - pos2) ** 2 for pos1, pos2 in zip(ship.position, asteroid.position)]))
                     # If collision occurs
                    if dist < (ship.radius + asteroid.radius):
                        # Ship destruct function. Add one to asteroids_hit
                        ship.asteroids_hit += 1
                        ship.destruct(map_size=self.scenario.map_size)
                        # Asteroid destruct function and mark for removal
                        asteroids.extend(asteroid.destruct(impactor=ship))
                        asteroid_remove_idxs.append(idx_ast)
                        print("collision")
                        check -= 10
                        # Stop checking this ship's collisions
                        break
        # Cull ships if not alive and asteroids that are marked for removal
        liveships = [ship for ship in liveships if ship.alive]
        asteroids = [asteroid for idx, asteroid in enumerate(asteroids) if idx not in asteroid_remove_idxs]
        # --- Check ship-ship collisions ---
        for ship1 in liveships:
            for ship2 in liveships:
                if (ship1 is not ship2) and (not ship2.is_respawning) and (not ship1.is_respawning):
                    dist = np.sqrt(sum([(pos1 - pos2) ** 2 for pos1, pos2 in zip(ship1.position, ship2.position)]))
                    if dist < ship1.radius + ship2.radius:
                        ship1.destruct(map_size=self.scenario.map_size)
                        ship2.destruct(map_size=self.scenario.map_size)
        # Cull ships that are not alive
        liveships = [ship for ship in liveships if ship.alive]
        # Update performance tracker with collisions timing
        if self.perf_tracker:
            perf_dict['collisions_check'] = time.perf_counter() - prev
            prev = time.perf_counter()
        # --- UPDATE SCORE CLASS -----------------------------------------------------------------------------------
        if self.perf_tracker:
            self.score.update(self.ships, self.sim_time, perf_dict['controller_times'])
        else:
            self.score.update(self.ships, self.sim_time)
        # Update performance tracker with score timing
        if self.perf_tracker:
            perf_dict['score_update'] = time.perf_counter() - prev
            prev = time.perf_counter()
        # --- UPDATE GRAPHICS --------------------------------------------------------------------------------------
        #graphics.update(self.score, self.ships, asteroids, bullets, mines)
        # Update performance tracker with graphics timing
        if self.perf_tracker:
            perf_dict['graphics_draw'] = time.perf_counter() - prev
            prev = time.perf_counter()
        # --- CHECK STOP CONDITIONS --------------------------------------------------------------------------------
        self.sim_time += self.time_step
        self.step += 1
        # No asteroids remain
        if not asteroids:
            stop_reason = StopReason.no_asteroids
        # No ships are alive
        elif not liveships:
            stop_reason = StopReason.no_ships
        # All live ships are out of bullets and no bullets are on map
        elif not sum([ship.bullets_remaining for ship in liveships]) and not len(bullets)>0 and self.scenario.stop_if_no_ammo:
            stop_reason = StopReason.out_of_bullets
        # Out of time
        elif self.sim_time > self.time_limit:
            stop_reason = StopReason.time_expired
        # --- FINISHING TIME STEP ----------------------------------------------------------------------------------
        # Get overall time step compute time
        if self.perf_tracker:
            perf_dict['total_frame_time'] = time.perf_counter() - step_start
            perf_list.append(perf_dict)
        # Hold simulation so that it runs at realtime ratio if specified, else let it pass
        if self.realtime_multiplier != 0:
            time_dif = time.perf_counter() - step_start
            while time_dif < (self.time_step/self.realtime_multiplier):
                time_dif = time.perf_counter() - step_start
        #state = torch.cat((state, torch.tensor([check])))
        #states.append(state)

        ############################################
        # Finalization after scenario has been run #
        ############################################

        # Close graphics display
        #print(game_state)

        # Finalize score class before returning
        self.score.finalize(self.sim_time, self.stop_reason, self.ships)
        return states, check, 0, 0, True

