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
    penalty = []
    sReward = []
    hReward = []
    hPenalty = []

    # Retrieve variables from csv
    with open(inputfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            tmp = row[3].split()
            reward.append(float(tmp[1]))
            tmp = row[4].split()
            penalty.append(float(tmp[1]))
            tmp = row[6].split()
            sReward.append(float(tmp[1]))
            tmp = row[8].split()
            hReward.append(float(tmp[1]))
            tmp = row[9].split()
            hPenalty.append(float(tmp[1]))

    assert len(reward) == len(sReward) == len(hReward)

    print("No Error agent: {}".format(np.count_nonzero(penalty)))
    print("No Error solver: {}".format(len(sReward) - np.count_nonzero(sReward)))
    print("No Error heuristic: {}".format(np.count_nonzero(hPenalty)))


if __name__ == "__main__":

   main(sys.argv[1:])
