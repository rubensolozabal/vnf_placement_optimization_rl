#-*- coding: utf-8 -*-
"""
    Placement using Neural Combinational Optimization.

    Author: Ruben Solozabal, PhD student at the University of the Basque Country (UPV-EHU) - Bilbao
    Date: June 2019
"""
import logging
import tensorflow as tf
from environment import *
from service_batch_generator import *
from agent import *
from config import *
from solver import *
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
import csv
import os
from first_fit import *

""" Globals """
DEBUG = True


def print_trainable_parameters():
    """ Calculate the number of weights """

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print('shape: ', shape, 'variable_parameters: ', variable_parameters)
        total_parameters += variable_parameters
    print('Total_parameters: ', total_parameters)


def calculate_reward(env, networkServices, placement, num_samples):
    """ Evaluate the batch of states into the environmnet """

    lagrangian = np.zeros(config.batch_size)
    penalty = np.zeros(config.batch_size)
    reward = np.zeros(config.batch_size)
    constraint_occupancy = np.zeros(config.batch_size)
    constraint_bandwidth = np.zeros(config.batch_size)
    constraint_latency = np.zeros(config.batch_size)

    reward_sampling = np.zeros(num_samples)
    constraint_occupancy_sampling = np.zeros(num_samples)
    constraint_bandwidth_sampling = np.zeros(num_samples)
    constraint_latency_sampling = np.zeros(num_samples)

    indices = np.zeros(config.batch_size)

    # Compute environment
    for batch in range(config.batch_size):
        for sample in range(num_samples):
            env.clear()
            env.step(networkServices.service_length[batch], networkServices.state[batch], placement[sample][batch])
            reward_sampling[sample] = env.reward
            constraint_occupancy_sampling[sample] = env.constraint_occupancy
            constraint_bandwidth_sampling[sample] = env.constraint_bandwidth
            constraint_latency_sampling[sample] = env.constraint_latency

        penalty_sampling = agent.lambda_occupancy * constraint_occupancy_sampling + agent.lambda_bandwidth * constraint_bandwidth_sampling + agent.lambda_latency * constraint_latency_sampling
        lagrangian_sampling = reward_sampling + penalty_sampling

        index = np.argmin(lagrangian_sampling)

        lagrangian[batch] = lagrangian_sampling[index]
        penalty[batch] = penalty_sampling[index]
        reward[batch] = reward_sampling[index]

        constraint_occupancy[batch] = constraint_occupancy_sampling[index]
        constraint_bandwidth[batch] = constraint_bandwidth_sampling[index]
        constraint_latency[batch] = constraint_latency_sampling[index]

        indices[batch] = index

    return lagrangian, penalty, reward, constraint_occupancy, constraint_bandwidth, constraint_latency, indices

if __name__ == "__main__":

    """ Log """
    logging.basicConfig(level=logging.DEBUG)  # filename='example.log'
    # DEBUG, INFO, WARNING, ERROR, CRITICAL

    """ Configuration """
    config, _ = get_config()

    """ Environment """
    env = Environment(config.num_cpus, config.num_vnfd, config.env_profile)

    """ Network service generator """
    vocab_size = config.num_vnfd + 1
    networkServices = ServiceBatchGenerator(config.batch_size, config.min_length, config.max_length, vocab_size)

    """ Agent """
    agent = Agent(config)

    """ Configure Saver to save & restore model variables """
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

    print("Starting session ...")

    with tf.Session() as sess:

        # Activate Tensorflow CLI debugger
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Activate Tensorflow debugger in Tensorboard
        #sess = tf_debug.TensorBoardDebugWrapperSession(
        #    sess=sess,
        #    grpc_debug_server_addresses=['localhost:6064'],
        #    send_traceback_and_source_code=True)

        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Print total number of parameters
        print_trainable_parameters()

        # Learn model
        if config.learn_mode:
            """
                Learning
            """

            # Restore model
            if config.load_model:
                saver.restore(sess, "{}/tf_placement.ckpt".format(config.load_from))
                print("\nModel restored.")


            # Summary writer
            writer = tf.summary.FileWriter("summary/repo", sess.graph)

            if config.save_model:
                filePath = "{}/learning_history.csv".format(config.save_to)

                if not os.path.exists(os.path.dirname(filePath)):
                    os.makedirs(os.path.dirname(filePath))

                if os.path.exists(filePath) and not config.load_model:
                    os.remove(filePath)

            print("\nStart learning...")

            try:
                episode = 0
                for episode in range(config.num_epoch):

                    # New batch of states
                    networkServices.getNewState()

                    # Mask
                    mask = np.zeros((config.batch_size,config.max_length))
                    for i in range(config.batch_size):
                        for j in range(networkServices.service_length[i], config.max_length):
                            mask[i, j] = 1

                    # RL Learning
                    feed = {agent.input_: networkServices.state,
                            agent.input_len_: [item for item in networkServices.service_length],
                            agent.mask: mask}

                    # Run placement
                    placement, decoder_softmax, _, baseline = sess.run([agent.actor.decoder_exploration, agent.actor.decoder_softmax, agent.actor.attention_plot, agent.valueEstimator.value_estimate], feed_dict=feed)
                    # positions, attention_plot = sess.run([agent.actor.positions, agent.actor.attention_plot], feed_dict=feed)

                    # Interact with the environment to return reward
                    lagrangian, penalty, reward, constraint_occupancy, constraint_bandwidth, constraint_latency, indices = calculate_reward(env, networkServices, placement, 1)

                    placement_ = np.zeros((config.batch_size, config.max_length))
                    for batch in range(config.batch_size):
                        placement_[batch] = placement[int(indices[batch])][batch]

                    feed = {agent.placement_holder: placement_,
                            agent.input_: networkServices.state,
                            agent.input_len_: [item for item in networkServices.service_length],
                            agent.mask: mask,
                            agent.baseline_holder: baseline,
                            agent.lagrangian_holder: [item for item in lagrangian]}

                    # Update our value estimator
                    feed_dict_ve = {agent.input_: networkServices.state,
                                 agent.valueEstimator.target: lagrangian}

                    _, loss = sess.run([agent.valueEstimator.train_op, agent.valueEstimator.loss], feed_dict_ve)

                    # Update actor
                    summary, _, loss_rl = sess.run([agent.merged, agent.train_step, agent.loss_rl], feed_dict=feed)

                    # Print learning
                    if episode == 0 or episode % 100 == 0:

                        print("------------")
                        print("Episode: ", episode)
                        print("Minibatch loss: ", loss_rl)
                        print("Network service[batch0]: ", networkServices.state[0])
                        print("Len[batch0]", networkServices.service_length[0])
                        print("Placement[batch0]: ", placement_[0])

                        # agent.actor.plot_attention(attention_plot[0])
                        # print("prob:", decoder_softmax[0][0])
                        # print("prob:", decoder_softmax[0][1])
                        # print("prob:", decoder_softmax[0][2])

                        print("Baseline[batch0]: ", baseline[0])
                        print("Reward[batch0]: ", reward[0])
                        print("Penalty[batch0]: ", penalty[0])
                        print("Lagrangian[batch0]: ", lagrangian[0])

                        print("Value Estimator loss: ", np.mean(loss))
                        print("Mean penalty: ", np.mean(penalty))
                        print("Count_nonzero: ", np.count_nonzero(penalty))

                    if episode % 10 == 0:

                        # Save in summary
                        writer.add_summary(summary, episode)

                    if config.save_model and (episode == 0 or episode % 100 == 0):

                        # Save in csv
                        csvData = ['batch: {}'.format(episode),
                                   ' network_service[batch 0]: {}'.format(networkServices.state[0]),
                                   ' placement[batch 0]: {}'.format(placement_[0]),
                                   ' reward: {}'.format(np.mean(reward)),
                                   ' lagrangian: {}'.format(np.mean(lagrangian)),
                                   ' baseline: {}'.format(np.mean(baseline)),
                                   ' advantage: {}'.format(np.mean(lagrangian) - np.mean(baseline)),
                                   ' penalty: {}'.format(np.mean(penalty)),
                                   ' minibatch_loss: {}'.format(loss_rl)]

                        filePath = "{}/learning_history.csv".format(config.save_to)
                        with open(filePath, 'a') as csvFile:
                            writer2 = csv.writer(csvFile)
                            writer2.writerow(csvData)

                        csvFile.close()

                    # Save intermediary model variables
                    if config.save_model and episode % max(1, int(config.num_epoch / 5)) == 0 and episode != 0:
                        save_path = saver.save(sess, "{}/tmp.ckpt".format(config.save_to), global_step=episode)
                        print("\nModel saved in file: %s" % save_path)

                    episode += 1

                print("\nLearning COMPLETED!")

            except KeyboardInterrupt:
                print("\nLearning interrupted by user.")

            # Save model
            if config.save_model:
                save_path = saver.save(sess, "{}/tf_placement.ckpt".format(config.save_to))
                print("\nModel saved in file: %s" % save_path)

        else:
            """
                Inference
            """

            # New batch of states
            networkServices.getNewState()

            # Mask
            mask = np.zeros((config.batch_size, config.max_length))
            for i in range(config.batch_size):
                for j in range(networkServices.service_length[i], config.max_length):
                    mask[i, j] = 1

            # Count the number of models "m" used in the active search: model_1, model_2 ...
            m = 0
            while os.path.exists("{}_{}".format(config.load_from, m+1)):  m += 1

            placement_m = [[]] * m
            placement_temp_m = [[]] * m

            penalty_m = [0] * m
            penalty_temp_m = [0] * m

            lagrangian_m = [0] * m
            lagrangian_temp_m = [0] * m

            reward_m = [0] * m
            reward_temp_m = [0] * m

            constraint_occupancy_m = [0] * m
            constraint_occupancy_temp_m = [0] * m

            constraint_bandwidth_m = [0] * m
            constraint_bandwidth_temp_m = [0] * m

            constraint_latency_m = [0] * m
            constraint_latency_temp_m = [0] * m

            for i in range(m):
                # Restore variables from disk

                saver.restore(sess, "{}_{}/tf_placement.ckpt".format(config.load_from, i+1))
                print("Model restored.")

                # Compute placement
                feed = {agent.input_: networkServices.state,
                        agent.input_len_: [item for item in networkServices.service_length], agent.mask: mask}

                placement_temp, placement, decoder_softmax_temp, decoder_softmax = sess.run \
                    ([agent.actor.decoder_sampling, agent.actor.decoder_prediction, agent.actor.decoder_softmax_temp,
                      agent.actor.decoder_softmax], feed_dict=feed)

                # Interact with the environment with greedy placement
                lagrangian, penalty, reward, constraint_occupancy, constraint_bandwidth, constraint_latency, _ = calculate_reward(
                    env, networkServices, placement, 1)

                # Interact with the environment with sampling technique
                lagrangian_temp, penalty_temp, reward_temp, constraint_occupancy_temp, constraint_bandwidth_temp, constraint_latency_temp, indices = calculate_reward(
                    env, networkServices, placement_temp, agent.actor.samples)

                # Store the output of each model
                placement_m[i] = placement
                for batch in range(config.batch_size):
                    placement_temp_m[i].append(placement_temp[int(indices[batch])][batch])

                penalty_m[i] = penalty
                penalty_temp_m[i] = penalty_temp

                print("Errors model ", i, ":", np.count_nonzero(penalty_m[i]), "/", config.batch_size)
                print("Errors model temperature ", i, ":", np.count_nonzero(penalty_temp_m[i]), "/", config.batch_size)

                lagrangian_m[i] = lagrangian
                lagrangian_temp_m[i] = lagrangian_temp

                reward_m[i] = reward
                reward_temp_m[i] = reward_temp

                constraint_occupancy_m[i] = constraint_occupancy
                constraint_occupancy_temp_m[i] = constraint_occupancy_temp

                constraint_bandwidth_m[i] = constraint_bandwidth
                constraint_bandwidth_temp_m[i] = constraint_bandwidth_temp

                constraint_latency_m[i] = constraint_latency
                constraint_latency_temp_m[i] = constraint_latency_temp

            penalty_m = np.vstack(penalty_m)
            penalty_temp_m = np.vstack(penalty_temp_m)

            lagrangian_m = np.stack(lagrangian_m)
            lagrangian_temp_m = np.stack(lagrangian_temp_m)

            reward_m = np.stack(reward_m)
            reward_temp_m = np.stack(reward_temp_m)

            constraint_occupancy_m = np.stack(constraint_occupancy_m)
            constraint_occupancy_temp_m = np.stack(constraint_occupancy_temp_m)

            constraint_bandwidth_m = np.stack(constraint_bandwidth_m)
            constraint_bandwidth_temp_m = np.stack(constraint_bandwidth_temp_m)

            constraint_latency_m = np.stack(constraint_latency_m)
            constraint_latency_temp_m = np.stack(constraint_latency_temp_m)

            index = []

            best_placement = []
            best_placement_t = []
            best_lagrangian = []
            best_lagrangian_t = []
            best_penalty = []
            best_penalty_t = []
            best_reward = []
            best_reward_t = []
            best_constraint_occupancy = []
            best_constraint_occupancy_t = []
            best_constraint_bandwidth = []
            best_constraint_bandwidth_t = []
            best_constraint_latency = []
            best_constraint_latency_t = []

            # Calculate and store the best model
            for batch in range(config.batch_size):
                index_l = np.argmin([row[batch] for row in lagrangian_m])
                index_p = np.argmin([row[batch] for row in penalty_m])

                assert penalty_m[index_l][batch] <= penalty_m[index_p][batch]

                best_placement.append(placement_m[index_l][0][batch])
                best_lagrangian.append(lagrangian_m[index_l][batch])
                best_penalty.append(penalty_m[index_l][batch])
                best_reward.append(reward_m[index_l][batch])
                best_constraint_occupancy.append(constraint_occupancy_m[index_l][batch])
                best_constraint_bandwidth.append(constraint_bandwidth_m[index_l][batch])
                best_constraint_latency.append(constraint_latency_m[index_l][batch])

                # Temperature

                index_lt = np.argmin([row[batch] for row in lagrangian_temp_m])
                index_pt = np.argmin([row[batch] for row in penalty_temp_m])

                best_placement_t.append(placement_temp_m[index_l][batch])
                best_lagrangian_t.append(lagrangian_temp_m[index_l][batch])
                best_penalty_t.append(penalty_temp_m[index_l][batch])
                best_reward_t.append(reward_temp_m[index_l][batch])
                best_constraint_occupancy_t.append(constraint_occupancy_temp_m[index_l][batch])
                best_constraint_bandwidth_t.append(constraint_bandwidth_temp_m[index_l][batch])
                best_constraint_latency_t.append(constraint_latency_temp_m[index_l][batch])

            print("Total errors: ", np.count_nonzero(best_penalty), "/", config.batch_size)
            print("Total errors temperature: ", np.count_nonzero(best_penalty_t), "/", config.batch_size)

            # Test Gecode solver
            if config.enable_performance:

                sReward = np.zeros(config.batch_size)

                filePath = '{}_test.csv'.format(config.load_from)
                if os.path.exists(filePath):
                    os.remove(filePath)

                for batch in tqdm(range(config.batch_size)):


                    hPlacement, hEnergy, hCst_occupancy, hCst_bandwidth, hCcst_latency = first_fit(networkServices.state[batch], networkServices.service_length[batch], env)


                    hPenalty = agent.lambda_occupancy * hCst_occupancy + agent.lambda_bandwidth * hCst_bandwidth + agent.lambda_latency * hCcst_latency
                    hLagrangian = hEnergy + hPenalty


                    sPlacement, sSvc_bandwidth, sSvc_net_latency, sSvc_cpu_latency, sEnergy, sOccupancy, sLink_used = \
                        solver(networkServices.state[batch], networkServices.service_length[batch], env)

                    if sPlacement == None:
                        sReward[batch] = 0
                    else:
                        env.clear()
                        env.step(networkServices.service_length[batch], networkServices.state[batch], sPlacement)

                        assert sSvc_bandwidth == env.bandwidth
                        assert sSvc_net_latency == env.link_latency
                        assert sSvc_cpu_latency == env.cpu_latency
                        assert sEnergy == env.reward
                        assert sOccupancy == list(env.cpu_used)
                        assert sLink_used == list(env.link_used)

                        sReward[batch] = env.reward

                    # Print testing
                    print("solver reward: ", sReward[batch])
                    print("reward: ", best_reward_t[batch])
                    print("cstr_occupancy: ", best_constraint_occupancy_t[batch])
                    print("cstr_bw: ", best_constraint_bandwidth_t[batch])
                    print("cstr_lat: ", best_constraint_latency_t[batch])


                    # Save in test.csv
                    csvData = [' batch: {}'.format(batch),
                               ' network_service: {}'.format(networkServices.state[batch]),
                               ' placement: {}'.format(best_placement_t[batch]),
                               ' reward: {}'.format(best_reward_t[batch]),
                               ' penalty: {}'.format(best_penalty_t[batch]),
                               ' solver_placement: {}'.format(sPlacement),
                               ' solver_reward: {}'.format(sReward[batch]),
                               ' heuristic_placement: {}'.format(hPlacement),
                               ' heuristic_reward: {}'.format(hEnergy),
                               ' heuristic_penalty: {}'.format(hPenalty)]

                    filePath = '{}_test.csv'.format(config.load_from)
                    with open(filePath, 'a') as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerow(csvData)

                    csvFile.close()
