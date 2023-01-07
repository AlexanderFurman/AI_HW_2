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



def is_hidden(state, agent_id, pawn):
    pawn_location = gge.find_curr_location(state, pawn, agent_id)
    for key, value in state.player1_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    for key, value in state.player2_pawns.items():
        if np.array_equal(value[0], pawn_location) and gge.size_cmp(value[1], state.player1_pawns[pawn][1]) == 1:
            return True
    return False



def dumb_heuristic2(state, agent_id):
    sum_pawns = 0
    if agent_id == 0:
        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1
    if agent_id == 1:
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                sum_pawns += 1

    return sum_pawns


def points_calculator(occurences):
    if (occurences == 0):
        return 0
    if (occurences == 1):
        return 1
    if (occurences == 2):
        return 10
    if (occurences == 3):
        return 100


def points(pawnLocationList):
    coordinates = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    totalPoints = 0
    i = 0
    firstDiagonalOccurence = 0
    secondDiagonalOccurence = 0
    rowOccurence = 0
    columnOccurence = 0
    while (i != 3):
        for coord in coordinates:
            if (coord[0] == i):  # going over rows
                if (coord in pawnLocationList):
                    rowOccurence += 1
            if (coord[1] == i):  # going over columns
                if (coord in pawnLocationList):
                    columnOccurence += 1
        totalPoints += points_calculator(rowOccurence)
        totalPoints += points_calculator(columnOccurence)
        rowOccurence = 0
        columnOccurence = 0
        i += 1

    # going over diagonals
    for coord in coordinates:
        if (coord == (0, 0) or coord == (1, 1) or coord == (2, 2)):
            if (coord in pawnLocationList):
                firstDiagonalOccurence += 1
        if (coord == (0, 2) or coord == (1, 1) or coord == (2, 0)):
            if (coord in pawnLocationList):
                secondDiagonalOccurence += 1
    totalPoints += (points_calculator(firstDiagonalOccurence) + points_calculator(secondDiagonalOccurence))
    return totalPoints


def opponentFinnaWin(pawnLocationList, rivalPawnLocationList, state, agent_id):
    coordinates = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    myRowOccurence = 0
    myColumnOccurence = 0
    rivalRowOccurence = 0
    rivalColumnOccurence = 0
    i = 0
    while (i != 3):
        for coord in coordinates:
            if (coord[0] == i):  # going over rows
                if (coord in pawnLocationList):
                    myRowOccurence += 1
                if (coord in rivalPawnLocationList):
                    rivalRowOccurence += 1
            if (coord[1] == i):  # going over columns
                if (coord in pawnLocationList):
                    myColumnOccurence += 1
                if (coord in rivalPawnLocationList):
                    rivalColumnOccurence += 1
        if (rivalRowOccurence == 2 and myRowOccurence == 0):
            return True
        if (rivalColumnOccurence == 2 and myColumnOccurence == 0):
            return True
        rivalRowOccurence = 0
        myRowOccurence = 0
        rivalColumnOccurence = 0
        myColumnOccurence = 0
        i += 1
    myFirstDiagonalOccurence = 0
    mySecondDiagonalOccurence = 0
    rivalFirstDiagonalOccurence = 0
    rivalSecondDiagonalOccurence = 0

    for coord in coordinates:
        if (coord == (0, 0) or coord == (1, 1) or coord == (2, 2)):
            if (coord in pawnLocationList):
                myFirstDiagonalOccurence += 1
            if (coord in rivalPawnLocationList):
                rivalFirstDiagonalOccurence += 1
        if (coord == (0, 2) or coord == (1, 1) or coord == (2, 0)):
            if (coord in pawnLocationList):
                mySecondDiagonalOccurence += 1
            if (coord in rivalPawnLocationList):
                rivalSecondDiagonalOccurence += 1
        if (rivalFirstDiagonalOccurence == 2 and myFirstDiagonalOccurence == 0):
            return True
        if (rivalSecondDiagonalOccurence == 2 and mySecondDiagonalOccurence == 0):
            return True
    return False


def calc_pawn_value(state, agent_id):
    pawns = {}
    count = 0

    if agent_id == 0:
        pawns = state.player1_pawns
    else:
        pawns = state.player2_pawns

    for _, v in pawns.items():
        letter = v[1]
        on_board = (not np.array_equal(v[0], np.array([-1, -1])))

        if on_board:
            if letter == "B":
                count += 0.5
            elif letter == "M":
                count += 0.2
            else:
                count += 0.1
    return count


def smart_heuristic(state, agent_id):
    pawnLocationList = []
    rivalPawnLocationList = []
    if agent_id == 0:
        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                pawnLocationList.append((value[0][0], value[0][1]))
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id + 1, key):
                rivalPawnLocationList.append((value[0][0], value[0][1]))
    if agent_id == 1:
        for key, value in state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id, key):
                pawnLocationList.append((value[0][0], value[0][1]))
        for key, value in state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(state, agent_id - 1, key):
                rivalPawnLocationList.append((value[0][0], value[0][1]))
    myPoints = points(pawnLocationList)
    rivalPoints = points(rivalPawnLocationList)
    if (rivalPoints >= 100):  # opponent won
        return 0
    if (myPoints >= 100):
        return 1000
    if (opponentFinnaWin(pawnLocationList, rivalPawnLocationList, state,
                         1 - agent_id) and myPoints < 100):  # aka opponent about to win and we didn't
        return min(myPoints, 1)
    return myPoints + calc_pawn_value(state, agent_id)


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
    state1_p1 = np.array([v[0] for k, v in state_1.player1_pawns.items()])
    state2_p1 = np.array([v[0] for k, v in state_2.player1_pawns.items()])

    state1_p2 = np.array([v[0] for k, v in state_1.player2_pawns.items()])
    state2_p2 = np.array([v[0] for k, v in state_2.player2_pawns.items()])

    return (state1_p1 == state2_p1).all() and (state1_p2 == state2_p2).all()


class MinMaxNode:
    def __init__(self):
        self.parent = None
        self.sons = []
        self.value = None
        self.depth = None
        self.state = None
        self.action = None


def update_values(node):
    if len(node.sons) != 0:
        if node.depth % 2 == 0:
            value = float('-inf')
            for son in node.sons:
                update_values(son)
                if son.value > value:
                    value = son.value
                    node.value = value
        else:
            value = float('inf')
            for son in node.sons:
                update_values(son)
                if son.value < value:
                    value = son.value
                    node.value = value

def rb_heuristic_min_max(curr_state, agent_id, time_limit):
    start_time = time.time()
    agent_number = agent_id
    cushion = 2  # [s], stopping recursion 2 seconds before time limit to allow for exiting the nested recursion

    def rb_minimax_recursion(current_state, turn_id, current_depth):
        if time_elapsed(start_time) > time_limit - cushion:
            return -np.inf

        if current_depth == 0:
            return smart_heuristic(current_state, agent_id)

        if gge.is_final_state(current_state):
            return smart_heuristic(current_state, agent_id)

        neighbor_list = current_state.get_neighbors()

        if turn_id == agent_number:
            max_value = -np.inf

            for neighbor in neighbor_list:
                v = rb_minimax_recursion(neighbor[1], 1 - turn_id, current_depth - 1)
                max_value = max(v, max_value)
            return max_value

        else:
            min_value = +np.inf

            for neighbor in neighbor_list:
                v = rb_minimax_recursion(neighbor[1], 1 - turn_id, current_depth - 1)
                min_value = min(v, min_value)
            return min_value

    depth = 0
    neighbour_list = curr_state.get_neighbors()
    max_heuristic = -np.inf
    max_neighbour = None

    while (time.time() < start_time + time_limit - 2):
        for neighbour in neighbour_list:
            current_heuristic = rb_minimax_recursion(neighbour[1], agent_id, depth)
            if current_heuristic > max_heuristic:
                max_heuristic = current_heuristic
                max_neighbour = neighbour
        depth += 1
    return max_neighbour[0]


def alpha_beta(curr_state, agent_id, time_limit):
    start_time = time.time()
    agent_number = agent_id
    cushion = 1  # [s], stopping recursion 1 second before time limit to allow for exiting the nested recursion

    def rb_minimax_recursion(current_state, turn_id, current_depth, alpha, beta):
        if time_elapsed(start_time) > time_limit - cushion:
            return -np.inf

        if current_depth == 0:
            return smart_heuristic(current_state, agent_id)

        if gge.is_final_state(current_state):
            return smart_heuristic(current_state, agent_id)

        neighbor_list = current_state.get_neighbors()

        if turn_id == agent_number:
            max_value = -np.inf

            for neighbor in neighbor_list:
                v = rb_minimax_recursion(neighbor[1], 1 - turn_id, current_depth - 1, alpha, beta)
                max_value = max(v, max_value)
                alpha = max(alpha, max_value)
                if (max_value >= beta):
                    return np.inf
            return max_value

        else:
            min_value = +np.inf

            for neighbor in neighbor_list:

                v = rb_minimax_recursion(neighbor[1], 1 - turn_id, current_depth - 1, alpha, beta)
                min_value = min(v, min_value)
                beta = min(beta, min_value)
                if alpha >= min_value:
                    return -np.inf
            return min_value

    depth = 0
    neighbour_list = curr_state.get_neighbors()
    max_heuristic = -np.inf
    max_neighbour = None

    while (time.time() < start_time + time_limit - 1):
        for neighbour in neighbour_list:
            current_heuristic = rb_minimax_recursion(neighbour[1], agent_id, depth, -np.inf, +np.inf)
            if current_heuristic > max_heuristic:
                max_heuristic = current_heuristic
                max_neighbour = neighbour
        depth += 1
    return max_neighbour[0]


def probability_of_happening(curr_state, next_state, num_neighbors, agent_id):
    # this function is called only for the rival player. Therefore we are playing from his perspective, seeing
    # if we can take pawns of the rival player, aka our actual player
    p = 1 / num_neighbors
    myOldPawnLocationList = []
    rivalOldPawnLocationList = []
    myNewPawnLocationList = []
    rivalNewPawnLocationList = []

    # Here we are creating lists of our pawn locations on the board and our opponent's pawn locations on the board,
    # both before and after the move we make. This is in order to see what changed on the board
    if agent_id == 0:
        for key, value in curr_state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(curr_state, agent_id, key):
                myOldPawnLocationList.append((value[0][0], value[0][1], value[1]))
        for key, value in curr_state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(curr_state, agent_id + 1, key):
                rivalOldPawnLocationList.append((value[0][0], value[0][1], value[1]))
    if agent_id == 1:
        for key, value in curr_state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(curr_state, agent_id, key):
                myOldPawnLocationList.append((value[0][0], value[0][1], value[1]))
        for key, value in curr_state.player1_pawns.items():
            if (not np.array_equal(value[0], not_on_board)) and (not is_hidden(curr_state, agent_id - 1, key)):
                rivalOldPawnLocationList.append((value[0][0], value[0][1], value[1]))
    if agent_id == 0:
        for key, value in next_state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(next_state, agent_id, key):
                myNewPawnLocationList.append((value[0][0], value[0][1], value[1]))
        for key, value in next_state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(next_state, agent_id + 1, key):
                rivalNewPawnLocationList.append((value[0][0], value[0][1], value[1]))
    if agent_id == 1:
        for key, value in next_state.player2_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(next_state, agent_id, key):
                myNewPawnLocationList.append((value[0][0], value[0][1], value[1]))
        for key, value in next_state.player1_pawns.items():
            if not np.array_equal(value[0], not_on_board) and not is_hidden(next_state, agent_id - 1, key):
                rivalNewPawnLocationList.append((value[0][0], value[0][1], value[1]))

    # now we are going to check if the move included capturing an opponent's pawn
    captured = False
    for pawn in rivalOldPawnLocationList:
        if pawn not in rivalNewPawnLocationList:  # meaning the pawn was there and now isn't, aka captured
            p *= 2
            captured = True
            break

    if not captured:  # checking if captures own pawn
        for pawn in myOldPawnLocationList:  # for every pawn in my old list
            if pawn not in myNewPawnLocationList:  # if it isn't in the new list
                if pawn[2] == 'S':  # if it's small
                    if ((pawn[0], pawn[1], 'B') in myNewPawnLocationList or (
                    pawn[0], pawn[1], 'M') in myNewPawnLocationList):  # if the new pawn in this location is bigger
                        captured = True
                        p *= 2
                        break
                if pawn[2] == 'M':  # if it's medium
                    if ((pawn[0], pawn[1], 'B') in myNewPawnLocationList):  # if the new pawn in this location is bigger
                        captured = True
                        p *= 2
                        break

    # now we are going to check if the move included the moving/placing of a SMALL pawn

    if not captured:
        # in order to check the case of a small pawn appearing on the board at a new location, there are two options:
        # either we moved it directly (made a move with S), or it was unravelled by moving a bigger encapsulating pawn
        # (in which case we did not make a move with S)
        small_changed = False
        for pawn in myNewPawnLocationList:
            if (pawn[2] != 'S'):
                continue
            if pawn not in myOldPawnLocationList:
                p *= 2
                small_changed = True
                break
        # here we are checking if the small pawn appeared at a new spot because we moved a pawn that was encapsulating it
        # we check that by checking if any other of our pawns were moved/placed on the board. If so, then we will divide
        # the probability by 2
        if small_changed:
            for pawn in myNewPawnLocationList:
                if (pawn[2] == 'S'):
                    continue
                if pawn not in myOldPawnLocationList:
                    p /= 2
                    break

    return p


# here we make the sum of all probabilities equal to 1
def p_normalization(probabilities, num_neighbors):
    total_prob = 0
    normal_p_value = 1 / num_neighbors
    for prob in probabilities:
        total_prob += prob
    factorization = 1 / total_prob
    new_total_prob = 0
    for prob in probabilities:
        prob = prob * factorization
        new_total_prob += prob


def expectimax(curr_state, agent_id, time_limit):
    start_time = time.time()
    agent_number = agent_id
    cushion = 2  # [s], stopping recursion 2 seconds before time limit to allow for exiting the nested recursion

    def rb_expectimax_recursion(current_state, turn_id, current_depth):
        if time_elapsed(start_time) > time_limit - cushion:
            return -np.inf

        if current_depth == 0:
            return smart_heuristic(current_state, agent_id)

        if gge.is_final_state(current_state):
            return smart_heuristic(current_state, agent_id)

        neighbor_list = current_state.get_neighbors()

        if turn_id == agent_number:

            max_value = -np.inf

            for neighbor in neighbor_list:
                v = rb_expectimax_recursion(neighbor[1], 1 - turn_id, current_depth - 1)
                max_value = max(v, max_value)
            return max_value

        else:
            expected_value = 0
            max_probability = 0
            chosen_state = None
            probabilities = []
            num_neighbors = len(neighbor_list)
            for neighbor in neighbor_list:
                p = probability_of_happening(current_state, neighbor[1], num_neighbors, turn_id)
                probabilities.append(p)

            p_normalization(probabilities, num_neighbors)
            i = 0
            for neighbor in neighbor_list:
                v = rb_expectimax_recursion(neighbor[1], 1 - turn_id, current_depth - 1)
                expected_value = v
                expected_value += probabilities[i] * expected_value
                if (probabilities[i] > max_probability):
                    max_probability = probabilities[i]
                i += 1
            return expected_value

    depth = 0
    neighbour_list = curr_state.get_neighbors()
    max_heuristic = -np.inf
    max_neighbour = None

    while (time.time() < start_time + time_limit - 2):
        for neighbour in neighbour_list:
            current_heuristic = rb_expectimax_recursion(neighbour[1], agent_id, depth)
            if current_heuristic > max_heuristic:
                max_heuristic = current_heuristic
                max_neighbour = neighbour
        depth += 1
    return max_neighbour[0]
