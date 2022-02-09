from PIL import Image
from ple import PLE
from ple.games.flappybird import FlappyBird
import numpy as np



class NaiveAgent():
    def __init__(self, actions):
        self.actions = actions
    def pickAction(self, reward, obs):
        return self.actions[np.random.randint(0, len(self.actions))]
    
nb_frames = 1000
reward = 0.0
max_noops = 20

environment = PLE(FlappyBird(288,512,100), fps=30, display_screen=True,)
agents = NaiveAgent(environment.getActionSet())
environment.init()

# this slows the visual down
environment.force_fps = False
# lets do a random number of NOOP's ????
for i in range(np.random.randint(0, max_noops)):
    reward = environment.act(environment.NOOP)


# start our training loop - lol
for i in range(nb_frames):
    # if the game is over
    if environment.game_over():
        print(reward)
        environment.reset_game()

    obs = environment.getScreenRGB()
    action = agents.pickAction(reward, obs)
    reward += environment.act(action)

# def eval(genomes, configuration):
