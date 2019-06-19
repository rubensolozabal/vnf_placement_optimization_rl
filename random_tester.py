import numpy as np
from service_batch_generator import *
from environment import *
from config import *

if __name__ == "__main__":

    """ Configuration """
    config, _ = get_config()

    """ Environment """
    env = Environment(config.num_cpus, config.num_vnfd)

    """ Network service generator """
    vocab_size = config.num_vnfd + 1
    networkServices = ServiceBatchGenerator(1, config.min_length, config.max_length, vocab_size)

    # New batch of states
    networkServices.getNewState()

    placement = [0] * networkServices.service_length[0]

    try:
        e = 1
        goals = 0

        while True:

            for i in range(networkServices.service_length[0]):
                cpuID = np.random.randint(0, config.num_cpus-1,  dtype='int32')
                placement[i] = cpuID

            """ Place in the environment """
            env.clear()
            env.step(networkServices.service_length[0], networkServices.state[0], placement)
            reward = env.reward

            if env.invalid_placement == False and env.invalid_bandwidth == False and env.invalid_latency == False:
                goals += 1

            if e == 0 or e % 1000 == 0:

                print("\n-------------")
                print("Episode: ", e)
                print("NS: ", networkServices.state[0])
                print("Placement: ", placement)
                print("Goals: ", goals)
                print("Invalid placement:", env.invalid_placement)
                print("Invalid bandwidth:", env.invalid_bandwidth)
                print("Invalid latency:", env.invalid_latency)

            e += 1

    except KeyboardInterrupt:
        print("Interrupted by user.")
