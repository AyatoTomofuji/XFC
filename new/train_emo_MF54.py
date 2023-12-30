import math
import random
from TreeController import NewController2
from examples.XFC.input3Controller import input4Controller
from src.kesslergame import TrainerEnvironment, KesslerController
from Scenarios import *
from deap import creator, base, tools, algorithms
import numpy as np
import matplotlib.pyplot as plt

# both evolve
rng = np.random.default_rng()
child = 1

# dist, ang に speedを加えた
def running(gene, genes2, scenario):
    # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # creator.create("Individual", list, fitness=creator.FitnessMax)
    game = TrainerEnvironment()
    # Because of how the arcade library is implemented, there are memory leaks for instantiating the environment
    # too many times, keep instantiations to a small number and simply reuse the environment
    controllers = [input4Controller(gene, genes2), input4Controller(gene, genes2)]
    pre = time.perf_counter()
    score, perf_data = game.run(scenario=scenario, controllers=controllers)
    standard = np.array([0, 0.25, 0.5, 0.75, 1.0, 90])
    reason = score.stop_reason
    f1 = np.average([team.accuracy for team in score.teams])
    f2 = 1 - score.sim_time / 120 if reason == reason.no_asteroids else score.sim_time / 120

    # 両方最大化
    return f1, f2


best_out = np.random.uniform(-425, 425, size=150)


# 命中率と，メンバシップ関数の中央の変位


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

    def run_2out(gene, gene_out):
        accu, time = 0, 0
        for scene in Scenarios1:
            f1, f2 = running(gene, gene_out, scene)
            accu += f1
            time += f2
        accu /= len(Scenarios1)
        time /= len(Scenarios1)
        return accu, time


    def feasible(individual1):
        x1, x2, x3, x4, x5, x6 = individual1
        #return x1 < x2 < x3 < x4 < x5 and -0.2 < x1 < 1.0 and -0.2 < x2 < 1.0 and -0.2 < x3 < 1.0 and -0.2 < x4 < 1.0 and -0.2 < x5 < 1.0 and 0 <= x6 <= 1.0
        return

    # 制約適応度
    def constrained_fitness(gene):
        return run_2out(gene, best_out)


    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    dim = 6
    toolbox.register("attr_float", random.uniform, -0.2, 1.0)
    toolbox.register("attr_angle", random.uniform, 0.0, 1.0)
    toolbox.register("attr_speed", random.uniform, 0.0, 1.0)


    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_float, toolbox.attr_float, toolbox.attr_float,
                      toolbox.attr_float, toolbox.attr_float, toolbox.attr_angle, toolbox.attr_speed,
                      toolbox.attr_speed,toolbox.attr_speed, toolbox.attr_speed,toolbox.attr_speed, toolbox.attr_speed), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", constrained_fitness)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=-0.2, up=1.0, eta=20.0)

    toolbox.register("mutates", tools.mutPolynomialBounded, up=1, low=-0.2, indpb=1 / 6, eta=20)

    toolbox.register("select", tools.selNSGA2)

    standard = np.array([0, 0.25, 0.5, 0.75, 1.0, 90])
    NGEN = 10000  # 繰り返し世代数
    MU = 100  # 集団内の個体数
    CXPB = 0.9  # 交叉率
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.average, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # 第一世代の生成
    pop = toolbox.population(n=MU)
    pop_init = pop[:]
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]

    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    print(invalid_ind)

    print(f"{pop}\n{len(pop)}")

    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)

    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)


    def plots(gen, pop, stats):
        print(pop, stats)
        print(tools.ParetoFront().items)
        fitnesses = np.array([list(pop[i].fitness.values) for i in range(len(pop))])
        plt.plot(fitnesses[:, 0], fitnesses[:, 1], "r.", label="Optimized")
        plt.title(f"Fitnesses3 at gen{gen}")
        plt.xlabel("Hit Accuracy")
        plt.ylabel("Sum of Displacement")
        plt.grid(True)
        plt.show()

    # 最適計算の実行
    for gen in range(1, NGEN+1):
        # 子母集団生成
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        # 交叉と突然変異

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            # 交叉させる個体を選択
            print(ind1, ind2)
            if random.random() <= CXPB:
                # 交叉
                toolbox.mate(ind1, ind2)
            # 突然変異
            toolbox.mutates(ind1)
            toolbox.mutates(ind2)
            print(ind1, ind2)

            # 交叉と突然変異させた個体は適応度を削除する
            del ind1.fitness.values, ind2.fitness.values
        # 適応度を削除した個体について適応度の再評価を行う
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 次世代を選択
        pop = toolbox.select(pop + offspring, MU)
        print(f"Generation {gen}:")
        non_dom = tools.sortNondominated(pop, k=len(pop), first_front_only=True)
        #for j, ind in enumerate(non_dom[0]):
        #    print("Individual ", j + 1, ": ", ind, "Fitness: ", ind.fitness.values)

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

        if gen % 50 == 0:
            plots(gen, pop, stats)




    # 最終世代のハイパーボリュームを出力
