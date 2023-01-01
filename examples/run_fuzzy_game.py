import time
import random
from src.kessler_game.controller import *
from src.kessler_game import Scenario, KesslerGame, GraphicsType
from Controller import NewController
from test_fuzzy_controllers import T1Controller, T2Controller


def timeout(input_data):
    # Function for testing the timing out by the simulation environment
    wait_time = random.uniform(0.02, 0.03)
    time.sleep(wait_time)

"""
class FuzzyController(ControllerBase):
    @property
    def name(self) -> str:
        return "Example Controller Name"

    def actions(self, ships: Tuple[SpaceShip], input_data: Dict[str, Tuple]) -> None:
        timeout(input_data)

        for ship in ships:
            ship.turn_rate = random.uniform(ship.turn_rate_range[0]/2.0, ship.turn_rate_range[1])
            ship.thrust = random.uniform(ship.thrust_range[0], ship.thrust_range[1])
            ship.fire_bullet = random.uniform(0.45, 1.0) < 0.5
"""


def CILrun():
    if __name__ == "__main__":
        # Available settings
        settings = {
            "frequency": 60,
            "real_time_multiplier": 2,
            "graphics_on": True,
            "sound_on": True,
            "prints": True,
            "full_dashboard": True
        }

        # Instantiate an instance of FuzzyAsteroidGame
        game = KesslerGame(settings=settings)

        scenario_ship = Scenario(name="Multi-Ship",
                                 num_asteroids=5,
                                 ship_states=[{"position": (300, 400), "angle": 0, "lives": 3, "team": 1},
                                              {"position": (700, 400), "angle": -10, "lives": 3, "team": 2},
                                              ],
                                 ammo_limit_multiplier=0.9)
        controllers = [NewController(), NewController()]
        pre = time.perf_counter()
        score, perf_data = game.run(scenario=scenario_ship, controllers=controllers)
        print('Scenario eval time: '+str(time.perf_counter()-pre))
        print(score.stop_reason)
        print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
        print('Deaths: ' + str([team.deaths for team in score.teams]))
        print('Accuracy: ' + str([team.accuracy for team in score.teams]))
        print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))
CILrun()
