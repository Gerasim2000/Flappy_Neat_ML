from ple import PLE
from ple.games.flappybird import FlappyBird
import numpy as np
import neat
import os
import warnings
import matplotlib.pyplot as plt
import time



iterations = 1000
alpha = 0.1
gamma = 1

class Agent():
    
    # Creates Q-table and sets initial parameters
    def __init__(self, game):
        
        # Game window is 256 x 512
        # With higher numbers -> more precise, but also slower
        bird_y = 26
        top_pipe = 26
        bottom_pipe = 26
        pipe_dist = 13
        # bird_speed = 10
        
        # The Q table consists of 4 dimensions(representing the game state) and the 2 actions
        self.Q_table = np.zeros((bird_y, top_pipe, bottom_pipe, pipe_dist, 2), dtype = int)
    
    # Receives game state, chooses best action
    def chooseAction(self, environment):
        
        # Handle Game State information
        game_state = environment.getGameState()
        bird_y = game_state["player_y"]
        top_pipe = abs(bird_y - game_state["next_pipe_top_y"]) 
        bottom_pipe = abs(bird_y - game_state["next_pipe_bottom_y"])
        # pipe_dist = abs(game_state["next_pipe_dist_to_player"])
        
        # Q_table action values for current state
        jump_value = self.Q_table[bird_y][top_pipe][bottom_pipe][0]
        noop_value = self.Q_table[bird_y][top_pipe][bottom_pipe][1]
        
        # performs best action
        if(jump_value >= noop_value):
            reward = environment.act(environment.getActionSet()[0])      # Jump   
        else:
            reward = environment.act(environment.getActionSet()[1])      # NOOP

        return reward
    


#Fitness function
def run():

    scores = []                                                     # all scores for current generation

    environment.init()
    for i in range(iterations):

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
