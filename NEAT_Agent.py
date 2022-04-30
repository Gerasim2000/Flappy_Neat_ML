from Environment.ple import PLE
from Environment.ple.games.flappybird import FlappyBird
import numpy as np
import neat
import os
import warnings
import matplotlib.pyplot as plt
import pickle
import time

# WARNING CHANGED INPUT SIZE TO 3


# Feed the game state into the Input Layer
# Return the activated output
def networkOutput(network, game):
    game_state = game.getGameState()
    
    # bird moves only on y axis (this is all we need for bird location)
    bird_y = game_state["player_y"]
    
    # horizontal distance to next pair of pipes
    pipe_dist = abs(game_state["next_pipe_dist_to_player"])
    
    # player velocity
    bird_speed = game_state["player_vel"]
    
    # vertical distance to both pipes ( currently testing only 1 pipe)
    top_pipe = abs(bird_y - game_state["next_pipe_top_y"])
    bottom_pipe = abs(bird_y - game_state["next_pipe_bottom_y"])
    next_pipe_bottom = game_state["next_next_pipe_bottom_y"]
    next_pipe_top = game_state["next_next_pipe_top_y"]
    
    # output = network.activate((bird_y, pipe_dist, bird_speed, top_pipe, bottom_pipe))
    output = network.activate((top_pipe, bottom_pipe, next_pipe_top, next_pipe_bottom, pipe_dist, bird_speed)) # give all inputs
    return output

scores = [] 

#Fitness function
def eval(genomes, configuration):
    global generation
    generation += 1
    
    print("===================== Generation "  + str(generation) + " =====================")
    
    # if(generation >= 80): 
    #     environment.force_fps = False
    #     environment.display_screen = True
    #     print(str(environment.getActionSet()))
    
                                                        # all scores for current generation

    environment.init()
    for genome_id, genome in genomes:

        net = neat.nn.FeedForwardNetwork.create(genome, configuration)    
        pipe_count = 0                                              # this is the score
        
        # Game loop for a single genome
        while True:  
            last_score = environment.game.getScore()                # score before action
            
            output = networkOutput(net, environment)                 # Get output from network
            if output[0] > 0.5:
                environment.act(environment.getActionSet()[0])      # Jump
                # print("Jump")
            else:
                environment.act(environment.getActionSet()[1])      # Don't jump (NOOP action)
                # print("NOOP")
            
            # if a pipe has passed (score should be > "positive" value in PLE initialization)
            score = environment.game.getScore() - last_score
            if(score >= 1): 
                # print("and anotha one: " + str(score))
                pipe_count += 1
            
            # when genome dies
            if(environment.lives() <= 0):
                genome.fitness = environment.game.score
                environment.init()
                print("Dead: " + str(genome_id) + " score: " + str(pipe_count))
                scores.append(pipe_count)
                break
            
def run(): 
    # Load NEAT configuration
    config_path = os.getcwd() + os.sep + "configuration.txt" # my config = configuration.txt
    configuration = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    # Initiate population and show statistics for each generation
    population = neat.Population(configuration)
    stats = neat.StatisticsReporter()
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(stats)
    # population.add_reporter(neat.Checkpointer(5))
    
    # to load a previous generation
    # population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-49')
    best = population.run(eval, 200)
    
    # plot_stats(stats)
    return best    
    

# suppress warnings on environment startup
warnings.filterwarnings("ignore")  
generation = 0
# Force fps = false for human speed
environment = PLE(FlappyBird(288,512,110), fps=30, display_screen=False, add_noop_action=True,
                      reward_values = {"positive": 2.0, "negative": -1.0, "tick": 0.01, "loss": -2.0, "win": 2.0}, 
                      force_fps=True)

start = time.time()

best = run()

end = time.time()

print("Time: ", end - start)

# with open('NEAT_trained', 'wb') as files:
    # pickle.dump(best, files)
# with open('Neat_Training_Data_1000', 'wb') as file:
    # print(scores)
    # pickle.dump(scores, file)    
    
