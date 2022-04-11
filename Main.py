from Environment.ple import PLE
from Environment.ple.games.flappybird import FlappyBird
import numpy as np
import pickle
import os
import neat

class Agent():

    def getQ(self):
        return self.Q_table
    
    def setQ(self, Q_trained):
        self.Q_table = Q_trained
    # Receives game state, chooses best action
    def chooseAction(self, environment, game_state):
        
        bottom_pipe = game_state[0]                                                                 # Choose action from game_state
        top_pipe = game_state[1]
        next_pipe_top = game_state[2]
        pipe_dist = game_state[3]
        bird_speed = game_state[4]
        
        # Q_table action values for current state
        jump_value = self.Q_table[bottom_pipe][top_pipe][next_pipe_top][pipe_dist][bird_speed][0]
        noop_value = self.Q_table[bottom_pipe][top_pipe][next_pipe_top][pipe_dist][bird_speed][1]
        
        # performs best action
        # print(str(jump_value) + " " + str(noop_value))
        if(jump_value <= noop_value):
            reward = environment.act(environment.getActionSet()[0])      # Jump
            # print("Jump")
            return int(1), reward 
        else:
            reward = environment.act(environment.getActionSet()[1])      # NOOP
            # print("NOOP")
            return int(0), reward

def shapeGameState(environment):
    
    game_state = environment.getGameState()

    # Create an exponential relative scale 
    bird_y = int(game_state["player_y"])
    pipe_dist = int(game_state["next_pipe_dist_to_player"])
    bird_speed = int(game_state["player_vel"])
    pipe1_bottom = int(bird_y - game_state["next_pipe_bottom_y"])
    
    # Reshape Game_State
    
    #                                                                                               First pair of pipes: bottom pipe (y coordinate)
    pipe1_bottom += 200
    if pipe1_bottom < 10:
        pipe1_bottom = 0
    elif(pipe1_bottom >= 400):
        pipe1_bottom = 19
    else:
        # Round to the nearest 10 (make the number fit between 1 and 18)
        pipe1_bottom = pipe1_bottom - (pipe1_bottom % 20)
        pipe1_bottom = pipe1_bottom // 20
     
    #                                                                                               First pair of pipes: top pipe (y coordinate)   
    pipe1_top = int(bird_y - game_state["next_pipe_top_y"])
    pipe1_top += 200
    if pipe1_top < 10:
        pipe1_top = 0
    elif(pipe1_top >= 400):
        pipe1_top = 19
    else:
        # Round to the nearest 10 (make the number fit between 1 and 18)
        pipe1_top = pipe1_top - (pipe1_top % 20)
        pipe1_top = pipe1_top // 20
        
    #                                                                                               Second pair of pipes: top pipe (y coordinate)
    pipe2_top = int(game_state["next_next_pipe_top_y"])
    if pipe2_top < 100:
        pipe2_top = 0
    elif(pipe2_top >= 300):
        pipe2_top = 7
    else:
        # Round to the nearest 10 (make the number fit between 1 and 18)
        pipe2_top = pipe2_top - (pipe2_top % 25)
        pipe2_top = pipe2_top // 25
    
    
    #                                                                                               Pipe distance (x coordinate)
    if pipe_dist < 239:
        pipe_dist = pipe_dist - (pipe_dist % 12)
        pipe_dist = pipe_dist // 12
    else:
        pipe_dist = 19

    #                                                                                               Bird velocity
    bird_speed += 10
    bird_speed = bird_speed - (bird_speed % 5)
    bird_speed = bird_speed // 5
    if(bird_speed > 3): bird_speed = 3
    
    # print("Pipe diff: ", bottom_pipe, " Bird: ", bird_y, " Bottom pipe: ", game_state["next_pipe_bottom_y"])
    # print("Speed: ", bird_speed)
    game_state_reshaped = [pipe1_bottom, pipe1_top, pipe2_top, pipe_dist, bird_speed]
    
    return game_state_reshaped
    
def runQ(iterations, environment):
    for i in range(iterations):
        
        Q_Agent = Agent()
        Q_Agent.setQ(np.load('q_trained.npy'))
        score_pipes = 0
        # Game loop for a single genome
        while True:  

            # HERE: Reshape the gameState Matrix (5 distance vectors)
            s = shapeGameState(environment=environment)

            # HERE: Call method chooseAction with the specified gameState
            action, reward = Q_Agent.chooseAction(environment = environment, game_state = s)

            # Keep track of score
            if(reward >= 1.0): score_pipes += 1

            # when genome dies
            if(environment.lives() <= 0):
                print("Generation: ", i, " Score: ", str(score_pipes))
                environment.init()
                # print("Try: " + str(i) + " score: " + str(math.floor(environment.game.score()/2)))
                break    

def networkOutput(network, environment):
    game_state = environment.getGameState()
    
    # bird moves only on y axis (this is all we need for bird location)
    bird_y = game_state["player_y"]
    
    # horizontal distance to next pair of pipes
    pipe_dist = abs(game_state["next_pipe_dist_to_player"])
    
    # player velocity
    bird_speed = game_state["player_vel"]
    
    # vertical distance to both pipes ( currently testing only 1 pipe)
    top_pipe = abs(bird_y - game_state["next_pipe_top_y"])
    bottom_pipe = abs(bird_y - game_state["next_pipe_bottom_y"])
    next_pipe_bottom = abs(bird_y - game_state["next_next_pipe_bottom_y"])
    next_pipe_top = abs(bird_y - game_state["next_next_pipe_top_y"])
    
    # output = network.activate((bird_y, pipe_dist, bird_speed, top_pipe, bottom_pipe))
    output = network.activate((top_pipe, bottom_pipe, next_pipe_top, next_pipe_bottom, pipe_dist, bird_speed)) # give all inputs
    return output

def runNeat(iterations, environment):
    with open('NEAT_trained', 'rb') as file:
        genome = pickle.load(file)
    config_path = os.getcwd() + os.sep + "configuration.txt" # my config = configuration.txt
    configuration = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)    
    Neat_agent = neat.nn.FeedForwardNetwork.create(genome, configuration)
    for i in range(iterations):
        pipe_count = 0
        while True:  
                
                reward = 0
                output = networkOutput(Neat_agent, environment)                 # Get output from network
                if output[0] > 0.5:
                    reward = environment.act(environment.getActionSet()[0])      # Jump
                    # print("Jump")
                else:
                    reward = environment.act(environment.getActionSet()[1])      # Don't jump (NOOP action)
                    # print("NOOP")

                # if a pipe has passed (score should be > "positive" value in PLE initialization)
                
                if(reward >= 1):   
                    # print("and anotha one: " + str(score))
                    pipe_count += 1

                # when genome dies
                if(environment.lives() <= 0):
                    genome.fitness = environment.game.score
                    environment.init()
                    print("Dead: " + str(i) + " score: " + str(pipe_count))
                    break
    
iterations = 50

environment = PLE(FlappyBird(288,512,110), fps=30, display_screen=True, add_noop_action=True,
                      reward_values = {"positive": 10.0, "negative": -100.0, "tick": 0.01, "loss": -2.0, "win": 2.0}, 
                      force_fps=True)

environment.init()
 
runQ(iterations, environment)
# runNeat(iterations, environment)





