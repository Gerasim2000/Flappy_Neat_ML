from Environment.ple import PLE
from Environment.ple.games.flappybird import FlappyBird
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pickle
import time



class Agent():
    
    # Creates Q-table and sets initial parameters
    def __init__(self):
        
        # Game window is 256 x 512
        # With higher numbers -> more precise, but also slower
        # State space
        bottom_pipe = 20                                                                            # Q_Table Dimensions
        top_pipe = 20
        pipe_dist = 20
        next_pipe_bot = 8
        bird_speed = 4
        
        # The Q table consists of 4 dimensions(representing the game state) and the 2 actions
        self.Q_table = np.zeros((bottom_pipe, top_pipe, next_pipe_bot, pipe_dist, bird_speed, 2), dtype = float)
    def getQ(self):
        
        return self.Q_table
    
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
    
iterations = 20000
alpha = 0.7
gamma = 1
epsilon = 0.0

#Fitness function
def run():

    scores = []                                               # all scores for current generation
    agent = Agent()
    environment.init()
    for i in range(iterations):

        # After 500 iterations show gameplay
        if(i > 50000):
            environment.display_screen = True
            environment.force_fps = False
            # np.save('q_trained', agent.getQ())
        score_pipes = 0
        # Game loop for a single genome
        while True:  
            
            # HERE: Reshape the gameState Matrix (3 distance vectors)
            s = shapeGameState(environment=environment)
                     
            # HERE: Call method chooseAction with the specified gameState
            action, reward = agent.chooseAction(environment = environment, game_state = s)
            
            s_new = shapeGameState(environment)
            #                                                                                                                           HERE: Update Q_Table
            current_Value = agent.Q_table[s[0]][s[1]][s[2]][s[3]][s[4]][action]
            future_Value = agent.Q_table[s_new[0]][s_new[1]][s_new[2]][s_new[3]][s_new[4]]
            
            agent.Q_table[s[0]][s[1]][s[2]][s[3]][s[4]][action] = current_Value + alpha * (reward + gamma * (np.max(future_Value)) - current_Value)
            
            # Keep track of score
            if(reward >= 1.0): score_pipes += 1
            
            # when genome dies
            if(environment.lives() <= 0):
                print("Generation: ", i, " Score: ", str(score_pipes))
                environment.init()
                scores.append(score_pipes)
                break
    return scores

# suppress warnings on environment startup
warnings.filterwarnings("ignore")  

# Force fps = false for human speed
environment = PLE(FlappyBird(288,512,110), fps=30, display_screen=False, add_noop_action=True,
                      reward_values = {"positive": 10.0, "negative": -100.0, "tick": 0.01, "loss": -2.0, "win": 2.0}, 
                      force_fps=True)

start = time.time()

results = run()

end = time.time()

print("Time: ", end - start)

# save the training data to a file for later visualisation
# with open('Q_Training_Data_50000', 'wb') as files:
    # pickle.dump(results, files)
