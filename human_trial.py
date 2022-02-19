from ple import PLE
from ple.games.flappybird import FlappyBird
import numpy as np
import neat
import os
import warnings
import matplotlib.pyplot as plt
import time

#Fitness function
def run():

    scores = []                                                     # all scores for current generation

    environment.init()
    for i in range(20):

        time.sleep(5)
        # Game loop for a single genome
        while True:  


            # when genome dies
            if(environment.lives() <= 0):
                environment.init()
                print("Try: " + str(i) + " score: " + str(environment.game.score()/2))
                break

    

# suppress warnings on environment startup
warnings.filterwarnings("ignore")  

# Force fps = false for human speed
environment = PLE(FlappyBird(288,512,110), fps=30, display_screen=True, add_noop_action=False,
                      reward_values = {"positive": 2.0, "negative": -1.0, "tick": 0.0, "loss": -2.0, "win": 2.0}, 
                      force_fps=False)
run()
