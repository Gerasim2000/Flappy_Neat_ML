from Environment.ple import PLE
from Environment.ple.games.flappybird import FlappyBird
import numpy as np
import neat
import os
import warnings
import matplotlib.pyplot as plt


# WARNING CHANGED INPUT SIZE TO 3

minMaxValues = {"player_vel" : 10, # -10 to 10 (make it 0 to 20)
                "pipe_dist" : 310, # 0 to 140 (beginning is 310)
                "bottom_pipe": 140} # 140 to 300
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
    
    diff = bird_y - game_state["next_pipe_bottom_y"]
    
    print("Speed: ", bird_speed, " X distance: ", pipe_dist, " Y bot pipe: ", game_state["next_pipe_bottom_y"], "height diff: ", diff)
    
    # output = network.activate((bird_y, pipe_dist, bird_speed, top_pipe, bottom_pipe))
    output = network.activate((bird_y, top_pipe, bottom_pipe)) # give all inputs
    return output



#Fitness function
def eval(genomes, configuration):
    global generation
    generation += 1
    
    print("===================== Generation "  + str(generation) + " =====================")
    
    if(generation >= 20): 
        environment.force_fps = False
        environment.display_screen = True
        print(str(environment.getActionSet()))
    
    scores = []                                                     # all scores for current generation

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
            else:
                environment.act(environment.getActionSet()[1])      # Don't jump (NOOP action)
            
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
                break

def plot_stats(stats):
    generation = range(len(stats.most_fit_genomes))
    highest_fitness = [c.fitness for c in stats.most_fit_genomes]
    avg_fitness = np.array(stats.get_fitness_mean())
    stdev_fitness = np.array(stats.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, highest_fitness, 'r-', label="best")

    plt.title("Average-highest fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    plt.show()

def run(): 
    # Load NEAT configuration
    config_path = os.getcwd() + os.sep + "configuration.txt" # my config = configuration.txt
    configuration = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    # Initiate population and show statistics for each generation
    population = neat.Population(configuration)
    stats = neat.StatisticsReporter()
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(5))
    
    # to load a previous generation
    # population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-49')
    best = population.run(eval, 50)
    
    plot_stats(stats)
    return best
    
    

# suppress warnings on environment startup
warnings.filterwarnings("ignore")  
generation = 0
# Force fps = false for human speed
environment = PLE(FlappyBird(288,512,110), fps=30, display_screen=False, add_noop_action=True,
                      reward_values = {"positive": 2.0, "negative": -1.0, "tick": 0.01, "loss": -2.0, "win": 2.0}, 
                      force_fps=True)
best = run()