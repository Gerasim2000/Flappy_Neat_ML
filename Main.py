from PIL import Image
from ple import PLE
from ple.games.flappybird import FlappyBird



p = PLE(FlappyBird(), fps=30, display_screen=True)
# agent = myAgentHere(allowed_actions=p.getActionSet())
p.init()


# def eval(genomes, configuration):
    