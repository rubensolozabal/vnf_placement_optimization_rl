#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys, getopt


def main(argv):

    # Input test file
    inputfile = ''

    try:
        opts, args = getopt.getopt(argv, "f:", ["file="])
    except getopt.GetoptError:
        print("test.py -f --file <testfile>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--file"):
            inputfile = arg

    reward = []
    constraints = []
    optReward = []

    # Retrieve variables from csv
    with open(inputfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            tmp = row[3].split()
            reward.append(float(tmp[1]))
            tmp = row[4].split()
            constraints.append(float(tmp[1]))
            tmp = row[6].split()
            optReward.append(float(tmp[1]))

    assert len(reward) == len(optReward)

    solution_not_found = 0
    only_nn_finds_solution = 0
    only_solver_finds_solution = 0

    nn_better = 0
    solver_better = 0
    equal_solution = 0

    cumulative_reward_opt = 0
    cumulative_reward = 0

    # Classify solution validity
    for i in range(len(reward)):

        if optReward[i] == 0 and constraints[i] > 0:
            solution_not_found += 1
        elif optReward[i] == 0:
            only_nn_finds_solution += 1
        elif constraints[i] > 0:
            only_solver_finds_solution += 1
        else:
            if optReward[i] < reward[i]:
                solver_better += 1
            elif reward[i] < optReward[i]:
                nn_better += 1
            else:
                equal_solution += 1

            cumulative_reward_opt += optReward[i]
            cumulative_reward += reward[i]

    # Print results
    print("No. total: ", len(reward))
    print("\nNone find a solution: ", solution_not_found)
    print("Only NN finds solution: ", only_nn_finds_solution)
    print("Only solver finds solution: ", only_solver_finds_solution)
    print("\nValid solution found in both cases: ", len(reward)-solution_not_found - only_nn_finds_solution - only_solver_finds_solution)
    print("     Solver is better: ", solver_better)
    print("     NN is better: ", nn_better)
    print("     Equal solution: ", equal_solution)
    print("     Performance: ", cumulative_reward / cumulative_reward_opt)

    # Plotting...
    fig, ax = plt.subplots()

    size = 0.3
    vals = np.array([[solution_not_found, only_nn_finds_solution, only_solver_finds_solution], [solver_better, nn_better, equal_solution]])

    cmap = plt.get_cmap("tab20c")
    outer_colors = cmap(np.arange(3) * 4)
    inner_colors = cmap(np.array([1, 2, 3, 5, 6, 7]))

    ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors, wedgeprops=dict(width=size, edgecolor='w'))
    ax.pie(vals.flatten(), radius=1 - size, colors=inner_colors, wedgeprops=dict(width=size, edgecolor='w'))
    ax.set(aspect="equal", title='Gecode vs NN')

    plt.show()


if __name__ == "__main__":

   main(sys.argv[1:])
