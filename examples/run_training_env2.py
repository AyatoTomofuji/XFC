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

        return [team.accuracy for team in score.teams]

    def run10(gene):
        min = 1.0
        for i in range(10):
            score = math.prod(running(gene))
            if min > score: min = score
        return min


    def blend(x_a, x_b, alpha):
        u = []
        for i in range(4):
            A = np.minimum(x_a[i], x_b[i]) - alpha * np.fabs(x_a[i] - x_b[i])
            B = np.maximum(x_a[i], x_b[i]) + alpha * np.fabs(x_a[i] - x_b[i])
            u.append(np.random.uniform(A, B))
        #u = np.clip(u, x_min, x_max)
        return u


    def tournament(_genes, _accu):
        a, b, d, e, f, g = rng.choice(len(_genes), 6, replace=False)
        if _accu[d] < _accu[e]:
            gene_a = genes[d]
        else:
            gene_a = genes[e]
        if _accu[f] < _accu[g]:
            gene_b = genes[f]
        else:
            gene_b = genes[g]
        return gene_a, gene_b


    standard = np.array([0, 100, 200, 90])
    genes = np.tile(standard, (10, 1))
    genes = (2 * np.random.rand(10, 4) - 1) * np.array([30, 30, 30, 10])+genes
    ave_list = []
    print(run10(genes[0]))
    accu = [run10(genes[j]) for j in range(len(genes))]

    for i in range(500):

        gene_a, gene_b = tournament(genes, accu)
        children = blend(gene_a, gene_b, 0.5)
        if run10(children) > np.min(accu):
            genes[np.argmin(accu)] = children
        accu = [run10(genes[j]) for j in range(len(genes))]

        first = np.argmax(accu)
        print(i)

        print(genes[first])


        with open("run10_worst3.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(accu)




