import game

from submission import heuristics_values
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    print("YAY LET'S START RUNNING STUFF")

    # PART1
    game.play_game("minimax","greedy_improved")
    print(heuristics_values)
    x_vals = np.linspace(1,len(heuristics_values), len(heuristics_values))
    print(x_vals)
    plt.plot(x_vals, heuristics_values)
    plt.xticks(x_vals)
    plt.title("Heuristic value of state chosen by our agent, over the game")
    plt.xlabel("Agent's turn number")
    plt.ylabel("Heuristic value")
    plt.show()



    # PART2
    # game.play_tournament("greedy", "random", 50)
    # game.play_tournament("greedy_improved", "random", 50)
    # game.play_tournament("greedy", "greedy_improved", 50)
