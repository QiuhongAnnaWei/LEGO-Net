'''
Python implementation of distances that are agnostic of model and training details.

Reference for Earthmover distance implementation: https://github.com/j2kun/earthmover
'''

import math
import torch
import numpy as np

from collections import Counter
from collections import defaultdict
from ortools.linear_solver import pywraplp
from sklearn.neighbors import NearestNeighbors


    
def euclidean_distance(x, y):
    """x, y: 1-dimensional array of any length"""
    return math.sqrt(sum((a - b)**2 for (a, b) in zip(x, y))) # each ele in zip: tuple


def chamfer_distance(x, y):
    """x and y: have dimension [numpt, featperpt], represent point clouds """
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(x)
    min_y_to_x = x_nn.kneighbors(y)[0] # closest 1 neighbor in x of each point in y
    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(y)
    min_x_to_y = y_nn.kneighbors(x)[0] # closest 1 neighbor in y of each point in x
    chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    return chamfer_dist

def chamfer_distance_batch(clean_input, per_input):
    """clean_input and per_input are of the dimensions [numscene, numobj, featperobj]"""
    chamfer_batch = []
    for x, y in zip(clean_input, per_input):
        chamfer_batch.append(1-chamfer_distance(x, y))
    return torch.tensor(chamfer_batch)


def earthmover_distance(p1, p2, p2attr=None, use_xy_key=True):
    '''
    Output the Earthmover distance between the two given points.
     - p1: an iterable of hashable iterables of numbers (list of tuples)
     - p2: an iterable of hashable iterables of numbers (list of tuples)
     - p2attr: additional attributes of p2 we want to parse and return, takes list form, 
               and p2[i] and p2attr[i] correspond to the same pt/entity in p2. If not given,
               returned assignment_attr is an empty dictionary.

    When p1 and p2 contain distinct tuples each appearing once, variable.solution_value=
    amount_to_move_x_y will all be 1, reducing the problem to a pure assignment problem
    (point in p1 to point in p2 will be bijective/1-to-1)

    Objective.value: sum of costs for each point's assignment. 
    '''
    # dist1 = {x: float(count) / len(p1) for (x, count) in Counter(p1).items()}
    # dist2 = {x: float(count) / len(p2) for (x, count) in Counter(p2).items()}
    dist1 = {x: float(count) for (x, count) in Counter(p1).items()} # moving dirt from
    dist2 = {x: float(count) for (x, count) in Counter(p2).items()} # moving dirt to
        # dist1:  {(0, 0): 1.0, (0, 1): 2.0, (0, -1): 1.0, (1, 0): 1.0, (-1, 0): 1.0}
    if (p2attr is not None):
        dist2attr = {p2[i]: p2attr[i] for i in range(len(p2))}
    
    solver = pywraplp.Solver('earthmover_distance', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    variables = dict()

    # for each pile in dist1, the constraint that says all the dirt must leave this pile
    dirt_leaving_constraints = defaultdict(lambda: 0)

    # for each hole in dist2, the constraint that says this hole must be filled
    dirt_filling_constraints = defaultdict(lambda: 0)

    # the objective
    objective = solver.Objective()
    objective.SetMinimization()

    for (x, dirt_at_x) in dist1.items():
        for (y, capacity_of_y) in dist2.items():
            amount_to_move_x_y = solver.NumVar(0, solver.infinity(), 'z_{%s, %s}' % (x, y))
            variables[(x, y)] = amount_to_move_x_y
            dirt_leaving_constraints[x] += amount_to_move_x_y
            dirt_filling_constraints[y] += amount_to_move_x_y
            objective.SetCoefficient(amount_to_move_x_y, euclidean_distance(x, y))
            # amount is like weight for each distance, =count of the tuple in input p1/p2

    # need to completely move all dirt, and fill all holes
    for x, linear_combination in dirt_leaving_constraints.items():
        solver.Add(linear_combination == dist1[x])
    for y, linear_combination in dirt_filling_constraints.items():
        solver.Add(linear_combination == dist2[y])

    status = solver.Solve()
    if status not in [solver.OPTIMAL, solver.FEASIBLE]:
        raise Exception('Unable to find feasible solution')
    
    var2AmountDistCost = dict()
    assignment = defaultdict(lambda: []) #{tuple(x) from p1 : [tuple(y1), tuple(y2) from p2]}
    assignment_attr = defaultdict(lambda: []) # {tuple(x) from p1 : [y1attr, y2attr]}

    for ((x, y), variable) in variables.items(): # x and y are input tuples, like (-1, 1)
        if variable.solution_value() != 0:  # variable.solution_value is amount_to_move_x_y
        
            assignment[x].append(y)
            if (p2attr is not None): assignment_attr[x].append(dist2attr[y])

            cost = euclidean_distance(x, y) * variable.solution_value() 
            
            if use_xy_key:
                var2AmountDistCost[(x, y)] = (variable.solution_value(), euclidean_distance(x, y), cost)
            else: # okay for bijective case
                var2AmountDistCost[(x)] = (variable.solution_value(), euclidean_distance(x, y), cost)
            # print("move {} dirt from {} to {} for a cost of {}".format(variable.solution_value(), x, y, cost))
    

    return assignment, assignment_attr, var2AmountDistCost, objective.Value()


def earthmover_assignment(p1, p2, p2attr=None):
    """ p2attr: if given has shape (numpt x any), and p2[i] and p2attr[i] correspond to same pt in p2.

        Return: p1_assignmemt = assigned destination for each point in p1 (in that exact order), which
                has shape (numpt x 2) (ex: [ [0,1], [2,2] ])
    """
    assignment, assignment_attr, _, _ = earthmover_distance(p1, p2, p2attr=p2attr, use_xy_key=True) # {tuple(x) : [ tuple(y) ]}
    p1_assignmemt, p1_assignmemt_attr = [], []
    for x in p1:
        y = assignment[x][0] # since it is 1-1 assignment
        p1_assignmemt.append(list(y))
        
        if (p2attr is not None):
            y_attr = assignment_attr[x][0] # p2attr[i]
            p1_assignmemt_attr.append(y_attr) # p1_assignmemt_attr: (numpt, p2attr[i]'s shape=2)

    return p1_assignmemt, p1_assignmemt_attr


if __name__ == "__main__":
    # p1 = [(1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (4,2), (6,1), (6,1) ]
    # p2 = [(2,1), (2,1), (3,2), (3,2), (3,2), (5,1), (5,1), (5,1), (5,1), (5,1), (5,1), (5,1), (7,1) ]
        # move 2.0 dirt from (1, 1) to (2, 1) for a cost of 2.0
        # move 3.0 dirt from (1, 1) to (3, 2) for a cost of 6.708203932499369
        # move 5.0 dirt from (1, 1) to (5, 1) for a cost of 20.0
        # move 1.0 dirt from (4, 2) to (5, 1) for a cost of 1.4142135623730951
        # move 1.0 dirt from (6, 1) to (5, 1) for a cost of 1.0
        # move 1.0 dirt from (6, 1) to (7, 1) for a cost of 1.0

        # {((1, 1), (2, 1)): (2.0, 1.0, 2.0), ((1, 1), (3, 2)): (3.0, 2.23606797749979, 6.708203932499369), ((1, 1), (5, 1)): (5.0, 4.0, 20.0), ((4, 2), (5, 1)): (1.0, 1.4142135623730951, 1.4142135623730951), ((6, 1), (5, 1)): (1.0, 1.0, 1.0), ((6, 1), (7, 1)): (1.0, 1.0, 1.0)}
        # 32.122417494872465


    p1 = [
        (0, 0),
        (0, 1),
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
    ]

    p2 = [
        (0, 0),
        (0, 2),
        (0, 2),
        (0, -2),
        (2, 0),
        (-2, 0),
    ]
        # move 1.0 dirt from (0, 0) to (0, 0) for a cost of 0.0
        # move 2.0 dirt from (0, 1) to (0, 2) for a cost of 2.0
        # move 1.0 dirt from (0, -1) to (0, -2) for a cost of 1.0
        # move 1.0 dirt from (1, 0) to (2, 0) for a cost of 1.0
        # move 1.0 dirt from (-1, 0) to (-2, 0) for a cost of 1.0
    # var2AmountDistCost, totalCost = earthmover_distance(p1, p2)
    # print()
    # print(var2AmountDistCost)
        # {((0, 0), (0, 0)): (1.0, 0.0, 0.0), ((0, 1), (0, 2)): (2.0, 1.0, 2.0), ((0, -1), (0, -2)): (1.0, 1.0, 1.0), ((1, 0), (2, 0)): (1.0, 1.0, 1.0), ((-1, 0), (-2, 0)): (1.0, 1.0, 1.0)}
    # print(totalCost)
        # 5.0


    # print(earthmover_assignment(p1,p2))
    # [(0, 0), (0, 2), (0, -2), (2, 0), (-2, 0)]

    


    # x = [[0,0], [0,0], [0,0]]
    # y = [[0,0], [1,1], [2,2]]
        # min_x_to_y= [[0.], [0.], [0.]]
        # min_y_to_x= [[0.], [1.41421356], [2.82842712]]
    
    # chamfer_distance(x, y)


    # target =[[ 0.01912169,  0.06895404] ,
    # [ 0.18032169,  0.06895404],
    # [ 0.34152169,  0.06895404],
    # [ 0.01912169, -0.35704596],
    # [ 0.18032169 ,-0.35704596],
    # [ 0.34152169, -0.35704596]]
    # source=[[ 0.04558555,  0.08104482],
    # [ 0.21851258,  0.0911653 ],
    # [ 0.04403377, -0.37534121],
    # [ 0.374111,    0.09642385],
    # [ 0.20074204, -0.35700575],
    # [ 0.36269286, -0.35695836]]
    # p1 = [tuple(chair) for chair in source]
    # p2 = [tuple(chair) for chair in target]
    # _, _, _, relative_emd_left = earthmover_distance(p1, p2) 

    # print(relative_emd_left)
