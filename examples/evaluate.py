import math
import random
#from deap import base, creator
from Controller import NewController
from src.kesslergame import TrainerEnvironment
from src.kesslergame import Scenario
import time
import numpy as np
import csv

rng = np.random.default_rng()
child = 1


if __name__ == "__main__":
    # Instantiate an instance of TrainerEnvironment.
    # The default settings should be sufficient, but check out the documentation for more information
    settings = {
        "graphics_on": True,
        "sound_on": False,
        # "frequency": 60,
        "real_time_multiplier": 200,
        # "lives": 3,
        # "prints": True,
        "allow_key_presses": False
    }
    def running(gene):
        #creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        #creator.create("Individual", list, fitness=creator.FitnessMax)
        game = TrainerEnvironment(settings=settings)
        scenario = Scenario(name="Multi-Ship",
                 num_asteroids=5,
                 ship_states=[{"position": (300, 400), "angle": 0, "lives": 3, "team": 1},
                              {"position": (700, 400), "angle": 0, "lives": 3, "team": 2},
                              ],
                 ammo_limit_multiplier=1.0)
        # Because of how the arcade library is implemented, there are memory leaks for instantiating the environment
        # too many times, keep instantiations to a small number and simply reuse the environment
        controllers = [NewController(gene), NewController(gene)]
        score , perf_data= game.run(controllers=controllers, scenario=scenario)

        """print('Scenario eval time: ' + str(time.perf_counter() - pre))
        print(score.stop_reason)
        print('Asteroids hit: ' + str([team.asteroids_hit for team in score.teams]))
        print('Deaths: ' + str([team.deaths for team in score.teams]))
        print('Accuracy: ' + str([team.accuracy for team in score.teams]))
        print('Mean eval time: ' + str([team.mean_eval_time for team in score.teams]))"""
        if sum([team.deaths for team in score.teams]) == 6:
            return 0.0

        return math.prod([team.accuracy for team in score.teams])






    gene = [0.0, 100.0, 200.0, 90.0]

    scores = [0.0 for i in range(500)]
    sum1 = 0.0
    for i in range(500):
        score = running(gene)
        print(i)
        print(score)
        scores[i] = score
    ave = np.average(scores)
    std = np.std(scores)
    print("平均:",ave)
    print("標準偏差:",std)






