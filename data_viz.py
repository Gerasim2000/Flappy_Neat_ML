from Environment.ple import PLE
from Environment.ple.games.flappybird import FlappyBird
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pickle


def splitList(list, chunkSize):
      
    # looping till length l
    for i in range(0, len(list), chunkSize): 
        yield list[i:i + chunkSize]

def plotStats(list):
    average = []
    stdev = []
    max = []
    for generation in list:
        
        print("Item: ", generation)
        average.append(np.mean(generation))
        stdev.append(np.std(generation))
        max.append(np.max(generation))

    
    generations = range(len(average))
    sum = [x + y for x, y in zip(average, stdev)]
    # print("Gens: ", generations, " St dev avg: ", sum)
    plt.plot(generations, average, 'b-', label="average")
    plt.plot(generations, sum, 'g-.', label="+1 st. dev.")
    plt.plot(generations, max, 'r-', label="best")

    plt.title("Average-highest Score")
    plt.xlabel("Generation(each unit represents 500 game attempts)")
    plt.ylabel("Score(pairs of pipes passed)")
    plt.grid()
    plt.legend(loc="best")
    plt.show()
    
with open('Q_Training_Data_20000', 'rb') as file:
    qData = pickle.load(file)
    print(qData)
    plotStats(splitList(qData, 500))
    
with open('Neat_Training_Data_500', 'rb') as file:
    neatData = pickle.load(file)
    print(neatData)
    wholeData = splitList(neatData, 100)
    plotStats(wholeData)
    