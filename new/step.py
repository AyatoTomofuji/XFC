from Scenarios import *
from kessler_game.src.kesslergame.kessler_game_step import Game_step
from Controller_neuro import Controller_neuro

game = Game_step(controllers=[Controller_neuro], scenario=accuracy_test_1)
for i in range(100):
    # sample inputs
    score, perf, states = game.run_step([480, 180, False, False, 10])
    print(score, states)