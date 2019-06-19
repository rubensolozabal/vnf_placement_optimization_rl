import matplotlib.pyplot as plt
import csv
import sys, getopt


def main(argv):

    #Input learning history file
    inputfile = ''

    try:
        opts, args = getopt.getopt(argv,"f:",["file="])
    except getopt.GetoptError:
        print("test.py -f --file <file>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-f", "--file"):
            inputfile = arg

    reward = []
    baseline = []
    advantage = []
    penalty = []
    loss_agent = []
    lagrangian = []

    # Retrieve variables from csv
    with open(inputfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            tmp = row[3].split()
            reward.append(float(tmp[1]))
            tmp = row[4].split()
            lagrangian.append(float(tmp[1]))
            tmp = row[5].split()
            baseline.append(float(tmp[1]))
            tmp = row[6].split()
            advantage.append(float(tmp[1]))
            tmp = row[7].split()
            penalty.append(float(tmp[1]))
            tmp = row[8].split()
            loss_agent.append(float(tmp[1]))

    # Plotting...
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(reward, label='energy')
    ax[0].plot(baseline, label='baseline')
    ax[0].plot(penalty, label='penalty')
    ax[0].plot(lagrangian, label='lagrangian')
    ax[0].legend()
    ax[0].set(ylabel='Cost', title='Learning history')
    ax[0].grid()

    ax[1].plot(loss_agent, label='loss agent')
    ax[1].grid()
    ax[1].set(xlabel='samples (x100)', ylabel='loss agent')

    plt.show()


if __name__ == "__main__":

    main(sys.argv[1:])






