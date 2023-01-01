from src.kessler_game.kessler_game import KesslerGame, Scenario, TrainerEnvironment
from Controller import NewController


settings = {
        "frequency": 30,
        "real_time_multiplier": 2,
        "graphics_on": True,
        "sound_on": False,
        "prints": True,
        "full_dashboard": True
    }
fitness=0
controllers = {NewController(), NewController()}
scenario_ship = Scenario(name="Multi-Ship",
                                 num_asteroids=5,
                                 ship_states=[{"position": (300, 400), "angle": 0, "lives": 3, "team": 1},
                                              {"position": (700, 400), "angle": 180, "lives": 3, "team": 2},
                                              ],
                                 ammo_limit_multiplier=0.9)
for b in range(0, 3):
    game = TrainerEnvironment(settings=settings)
    score = game.run(scenario=scenario_ship, controllers=controllers)
    if score.deaths == 0:
        fitness += 24
    else:
        fitness += 18 / score.deaths - 5
