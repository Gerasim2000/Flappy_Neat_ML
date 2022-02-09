from ple import PLE
from ple.games.flappybird import FlappyBird
import numpy as np
import neat
import os
      
# Setup game environment
# environment = PLE(FlappyBird(288,512,100), fps=30, display_screen=True, reward_values = {"positive": 1.0, "negative": -1.0, "tick": 0.1, "loss": -5.0, "win": 5.0})

# this slows the visual down
# environment.force_fps = False

# Feed the game state into the Input Layer
# Return the activated output
def networkOutput(network, game):
    game_state = game.getGameState()
    
    # bird moves only on y axis (this is all we need for bird location)
    bird_y = game_state["player_y"]
    
    # horizontal distance to next pair of pipes
    # pipe_dist = game_state["next_pipe_dist_to_player"] 
    
    # player velocity
    bird_speed = game_state["player_vel"]
    
    # vertical distance to both pipes ( currently testing only 1 pipe)
    top_pipe = abs(bird_y - game_state["next_pipe_top_y"])
    bottom_pipe = abs(bird_y - game_state["next_pipe_bottom_y"])
    
    output = network.activate((bird_y, bird_speed, top_pipe, bottom_pipe))
    # output = network.activate((bird_y, top_pipe, bottom_pipe)) # give all inputs
    return output



#Fitness function
def eval(genomes, configuration):
    print("Fitness function start")
    
    environment = PLE(FlappyBird(288,512,100), fps=30, display_screen=True,
                      reward_values = {"positive": 1.0, "negative": -1.0, "tick": 0.1, "loss": -5.0, "win": 5.0}, 
                      force_fps=False)
    # environment.init()
    # environment.game._setup()
    #environment.init()
    
    print("New Generation")
    environment.init()
    for genome_id, genome in genomes:
        
        # environment.init()
        
        net = neat.nn.FeedForwardNetwork.create(genome, configuration)    
        # genome.fitness = 0
        print("Bird number: " + str(genome_id))
        
        # Game loop
        while True:
            
            # get output from network
            output = networkOutput(net, environment)
            if output[0] > 0.5:
                print("about to jump: " + str(output[0]) + " frame: " + str(environment.getFrameNumber()))
                environment.act(environment.getActionSet()[0])
                # environment.game.player.flap()
            else:
                environment.act(environment.getActionSet()[1])
            # print("output: " + str(output) + "score: " + str(environment.game.score))
            
            genome.fitness = environment._getReward()
            
            # if dead
            if(environment.lives() <= 0):

                environment.init()
                print("Dead: " + str(genome_id) + " score: " + str(genome.fitness))
                break
        
        
   


def run(): 
    # Load NEAT configuration
    config_path = os.getcwd() + os.sep + "configuration.txt" # my config = configuration.txt
    configuration = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    # Initiate population and show statistics for each generation
    population = neat.Population(configuration)
    population.add_reporter(neat.StatisticsReporter())
    best = population.run(eval, 20)
    
    return best
     
run()