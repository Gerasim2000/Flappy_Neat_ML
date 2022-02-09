from ple import PLE
from ple.games.flappybird import FlappyBird
import numpy as np
import neat
import os


environment = PLE(FlappyBird(288,512,100), fps=30, display_screen=True, reward_values = {"positive": 1.0, "negative": -1.0, "tick": 0.1, "loss": -5.0, "win": 5.0})
environment.force_fps = False
# environment.game._setup()
environment.init()