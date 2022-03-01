from ple import PLE
from ple.games.flappybird import FlappyBird
import numpy as np
import neat
import os
import warnings
import matplotlib.pyplot as plt
import time
import math

class Agent():
    
    # Creates Q-table and sets initial parameters
    def __init__(self):
        
        # Game window is 256 x 512
        # With higher numbers -> more precise, but also slower
        bird_y = 26
        top_pipe = 26
        bottom_pipe = 26
        pipe_dist = 13
        # bird_speed = 10
        
        # The Q table consists of 4 dimensions(representing the game state) and the 2 actions
        self.Q_table = np.zeros((bird_y, top_pipe, bottom_pipe, pipe_dist, 1), dtype = float)
    
    # Receives game state, chooses best action
    def chooseAction(self, environment, game_state):
        
        bird_y = game_state[0]
        top_pipe = game_state[1]
        bottom_pipe = game_state[2]
        # pipe_dist = game_state[3]
        
        # Q_table action values for current state
        jump_value = self.Q_table[bird_y][top_pipe][bottom_pipe][0]
        noop_value = self.Q_table[bird_y][top_pipe][bottom_pipe][1]
        
        # performs best action
        print(str(jump_value) + " " + str(noop_value))
        if(jump_value <= noop_value):
            reward = environment.act(environment.getActionSet()[0])      # Jump
            return int(1), reward 
        else:
            reward = environment.act(environment.getActionSet()[1])      # NOOP
            return int(0), reward

def shapeGameState(environment):
    
    game_state = environment.getGameState()
    
    bird_y = math.ceil(game_state["player_y"] // 20)
    top_pipe = math.ceil(abs(bird_y - game_state["next_pipe_top_y"]) // 20)
    bottom_pipe = math.ceil(abs(bird_y - game_state["next_pipe_bottom_y"]) // 20)
    
    game_state_reshaped = [bird_y, top_pipe, bottom_pipe]
    
    return game_state_reshaped
    
iterations = 20000
alpha = 0.9
gamma = 1
epsilon = 0.1

#Fitness function
def run():

    scores = []                                                     # all scores for current generation
    agent = Agent()
    environment.init()
    for i in range(iterations):

        # After 500 iterations show gameplay
        if(i > 18500):
            environment.display_screen = True
            environment.force_fps = False
                
        # Game loop for a single genome
        while True:  
            
            # HERE: Reshape the gameState Matrix (3 distance vectors)
            s = shapeGameState(environment=environment)
                     
            # HERE: Call method chooseAction with the specified gameState
            action, reward = agent.chooseAction(environment = environment, game_state = s)
            
            s_new = shapeGameState(environment)
            # HERE: Update Q_Table
            current_Value = agent.Q_table[s[0]][s[1]][s[2]][action][0]
            future_Value = agent.Q_table[s_new[0]][s_new[1]][s_new[2]]
            
            agent.Q_table[s[0]][s[1]][s[2]][action] = current_Value + alpha * (reward + gamma * (np.max(future_Value)) - current_Value)
            
            # when genome dies
            if(environment.lives() <= 0):
                environment.init()
                # print("Try: " + str(i) + " score: " + str(math.floor(environment.game.score()/2)))
                break

    

# suppress warnings on environment startup
warnings.filterwarnings("ignore")  

# Force fps = false for human speed
environment = PLE(FlappyBird(288,512,110), fps=30, display_screen=False, add_noop_action=True,
                      reward_values = {"positive": 5.0, "negative": -100.0, "tick": 0.01, "loss": -2.0, "win": 2.0}, 
                      force_fps=True)
run()
