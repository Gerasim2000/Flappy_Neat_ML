from Environment.ple import PLE
from Environment.ple.games.flappybird import FlappyBird
import numpy as np
import warnings
import matplotlib.pyplot as plt

class Agent():
    
    # Creates Q-table and sets initial parameters
    def __init__(self):
        
        # Game window is 256 x 512
        # With higher numbers -> more precise, but also slower
        # State space
        bottom_pipe = 40                                                                            # Q_Table Dimensions
        pipe_dist = 10
        bird_speed = 10
        
        # The Q table consists of 4 dimensions(representing the game state) and the 2 actions
        self.Q_table = np.zeros((bottom_pipe, pipe_dist, bird_speed, 2), dtype = float)
    
    # Receives game state, chooses best action
    def chooseAction(self, environment, game_state):
        
        # bird_y = game_state[0]
        # top_pipe = game_state[1]
        bottom_pipe = game_state[0]                                                                 # Choose action from game_state
        pipe_dist = game_state[1]
        bird_speed = game_state[2]
        
        # Q_table action values for current state
        jump_value = self.Q_table[bottom_pipe][pipe_dist][bird_speed][0]
        noop_value = self.Q_table[bottom_pipe][pipe_dist][bird_speed][1]
        
        # performs best action
        # print(str(jump_value) + " " + str(noop_value))
        if(jump_value <= noop_value):
            reward = environment.act(environment.getActionSet()[0])      # Jump
            return int(1), reward 
        else:
            reward = environment.act(environment.getActionSet()[1])      # NOOP
            return int(0), reward


minMaxValues = {"player_vel" : 10, # -10 to 10 (make it 0 to 20)
                "pipe_dist" : 310, # 0 to 140 (beginning is 310)
                "bottom_pipe": 140} # 140(highest vertical position) to 300(lowest vertical position)
# velocity - {0 - 9} 
# X distance - {0 - 9}
# Y difference - {0 - 19} from -90 to +90

def shapeGameState(environment):
    
    game_state = environment.getGameState()

    # Create an exponential relative scale 
    bird_y = int(game_state["player_y"])
    bottom_pipe = int(bird_y - game_state["next_pipe_bottom_y"])
    pipe_dist = int(game_state["next_pipe_dist_to_player"])
    bird_speed = int(game_state["player_vel"])
    # Potentially add diff between next two pipes
    # Reshape Game_State
    bottom_pipe += 200
    if bottom_pipe < 10:
        bottom_pipe = 0
    elif(bottom_pipe > 400):
        bottom_pipe = 19
    else:
        # Round to the nearest 10 (make the number fit between 1 and 18)
        bottom_pipe = bottom_pipe - (bottom_pipe % 10)
        bottom_pipe = bottom_pipe // 10
    
    # Place the values in 10 intervals 
    if pipe_dist < 200:
        pipe_dist = pipe_dist - (pipe_dist % 20)
        pipe_dist = pipe_dist // 20
    else:
        pipe_dist = 9

    bird_speed += 10
    bird_speed = bird_speed // 2
    if(bird_speed > 9): bird_speed = 9
    
    print("Pipe diff: ", bottom_pipe, " Bird: ", bird_y, " Bottom pipe: ", game_state["next_pipe_bottom_y"])
    game_state_reshaped = [bottom_pipe, pipe_dist, bird_speed]
    
    return game_state_reshaped
    
iterations = 1000
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
        if(i > 900):
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
            current_Value = agent.Q_table[s[0]][s[1]][s[2]][action]
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
                      reward_values = {"positive": 10.0, "negative": -100.0, "tick": 0.01, "loss": -2.0, "win": 2.0}, 
                      force_fps=True)
run()
