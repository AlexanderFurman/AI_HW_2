import random

import numpy as np

import Gobblet_Gobblers_Env as gge

import time


not_on_board = np.array([-1, -1])
heuristics_values = []

# agent_id is which player I am, 0 - for the first player , 1 - if second player
def dumb_heuristic1(state, agent_id):
    is_final = gge.is_final_state(state)
    # this means it is not a final state
    if is_final is None:
        return 0
    # this means it's a tie
    if is_final is 0:
        return -1
    # now convert to our numbers the win
    winner = int(is_final) - 1
    # now winner is 0 if first player won and 1 if second player won
    # and remember that agent_id is 0 if we are first player  and 1 if we are second player won
    if winner == agent_id:
        # if we won
        return 1
    else:
        # if other player won
        return -1


# checks if a pawn is under another pawn
def is_hidden(state, agent_id, pawn):
    pawn_location = gge.find_curr_location(state, pawn, agent_id)
    for key, value in state.player1_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    for key, value in state.player2_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    return False


# count the numbers of pawns that i have that aren't hidden
def dumb_heuristic2(state, agent_id):
    sum_pawns = 0
    pawnLocationList = []
    if agent_id == 0:
        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1
                pawnLocationList.append((value[0][0],value[0][1]))
    if agent_id == 1:
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1
                pawnLocationList.append((value[0][0],value[0][1]))
    print("list",pawnLocationList)
    return sum_pawns


def points_calculator(occurences):
    if (occurences==0):
        return 0
    if (occurences==1):
        return 1
    if (occurences==2):
        return 10
    if (occurences==3):
        return 100

def smart_heuristic(state, agent_id):
    pawnLocationList = []
    if agent_id == 0:
        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                pawnLocationList.append((value[0][0],value[0][1]))
    if agent_id == 1:
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                pawnLocationList.append((value[0][0],value[0][1]))
    print(pawnLocationList)
    coordinates = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    totalPoints=0
    i=0
    firstDiagonalOccurence=0
    secondDiagonalOccurence=0
    rowOccurence = 0
    columnOccurence =0
    while (i != 3):
        for coord in coordinates:
            if (coord[0]==i): #going over rows
                if (coord in pawnLocationList):
                    rowOccurence+=1
            if (coord[1]==i): #going over columns
                if (coord in pawnLocationList):
                    columnOccurence+=1
        totalPoints+=points_calculator(rowOccurence)
        totalPoints += points_calculator(columnOccurence)
        rowOccurence=0
        columnOccurence=0
        i+=1

    #going over diagonals
    for coord in coordinates:
        if (coord == (0,0) or coord==(1,1) or coord==(2,2)):
            if (coord in pawnLocationList):
                firstDiagonalOccurence += 1
        if (coord == (0,2) or coord==(1,1) or coord==(2,0)):
            if (coord in pawnLocationList):
                secondDiagonalOccurence += 1
    totalPoints += (points_calculator(firstDiagonalOccurence) + points_calculator(secondDiagonalOccurence))
    print(totalPoints)
    return totalPoints


# IMPLEMENTED FOR YOU - NO NEED TO CHANGE
def human_agent(curr_state, agent_id, time_limit):
    print("insert action")
    pawn = str(input("insert pawn: "))
    if pawn.__len__() != 2:
        print("invalid input")
        return None
    location = str(input("insert location: "))
    if location.__len__() != 1:
        print("invalid input")
        return None
    return pawn, location


# agent_id is which agent you are - first player or second player
def random_agent(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    rnd = random.randint(0, neighbor_list.__len__() - 1)
    return neighbor_list[rnd][0]


# TODO - instead of action to return check how to raise not_implemented
def greedy(curr_state, agent_id, time_limit):
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = 0
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = dumb_heuristic2(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    return max_neighbor[0]


# TODO - add your code here
def greedy_improved(curr_state, agent_id, time_limit):
    global heuristics_values
    neighbor_list = curr_state.get_neighbors()
    max_heuristic = 0
    max_neighbor = None
    for neighbor in neighbor_list:
        curr_heuristic = smart_heuristic(neighbor[1], agent_id)
        if curr_heuristic >= max_heuristic:
            max_heuristic = curr_heuristic
            max_neighbor = neighbor
    heuristics_values.append(max_heuristic)
    return max_neighbor[0]

def time_elapsed(start_time):
    return time.time() - start_time

def states_equal(state_1, state_2):
    # print(f"items of player2 = {state_2.player1_pawns.items()}")
    state1_p1 = np.array([v[0] for k,v in state_1.player1_pawns.items()])
    state2_p1 = np.array([v[0] for k,v in state_2.player1_pawns.items()])

    state1_p2 = np.array([v[0] for k,v in state_1.player2_pawns.items()])
    state2_p2 = np.array([v[0] for k,v in state_2.player2_pawns.items()])

    return (state1_p1 == state2_p1).all() and (state1_p2 == state2_p2).all()


def rb_heuristic_min_max(curr_state, agent_id, time_limit):
    start_time = time.time()
    agent_number = agent_id
    cushion = 2 # [s], stopping recursion 2 seconds before time limit to allow for exiting the nested recursion
    # time_for_recursion = 0
    # time_elapsed = 0

    def rb_minimax_recursion(current_state, turn_id, current_depth):
        print(f"time elapsed = {time_elapsed(start_time)}")
        if time_elapsed(start_time) > time_limit - cushion:
            return False

        if current_depth == 0:
            return (current_state, smart_heuristic(current_state, agent_id))

        if gge.is_final_state(current_state):
            return (current_state, smart_heuristic(current_state, agent_id))


        neighbor_list = current_state.get_neighbors()

        if turn_id == agent_number:
            max_heuristic = -np.inf
            max_state = None

            for neighbor in neighbor_list:
                v = rb_minimax_recursion(neighbor[1], 1 - turn_id, current_depth-1)
                if v == False:
                    return False
                current_heuristic = v[1]
                if current_heuristic > max_heuristic:
                    max_heuristic = current_heuristic
                    max_state = neighbor[1]
            return (max_state, max_heuristic)

        else:
            min_heuristic = +np.inf
            min_state = None

            for neighbor in neighbor_list:
                v = rb_minimax_recursion(neighbor[1], 1 - turn_id, current_depth-1)
                if v == False:
                    return False
                current_heuristic = v[1]
                if current_heuristic < min_heuristic:
                    min_heuristic = current_heuristic
                    min_state = neighbor[1]
            return (min_state,min_heuristic)


    depth = 0
    deepest_fully_scanned_solution = None
    recursion_start_time = time.time()
    current_solution = rb_minimax_recursion(curr_state, agent_id, depth)

    while(current_solution is not False):
    # while(depth < 2):
        deepest_fully_scanned_solution = current_solution[0]
        # print(f" deppest scan = {deepest_fully_scanned_solution}")
        depth += 1
        current_solution = rb_minimax_recursion(curr_state, agent_id, depth)

    print(f"time elapsed = {time.time()-start_time}")
    chosen_step = [neighbour for neighbour in curr_state.get_neighbors() if states_equal(neighbour[1],deepest_fully_scanned_solution)][0][0]
    return chosen_step


def alpha_beta(curr_state, agent_id, time_limit):
    raise NotImplementedError()


def expectimax(curr_state, agent_id, time_limit):
    raise NotImplementedError()

# these is the BONUS - not mandatory
def super_agent(curr_state, agent_id, time_limit):
    raise NotImplementedError()
