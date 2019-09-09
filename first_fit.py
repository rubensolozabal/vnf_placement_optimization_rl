from environment import *
from service_batch_generator import *
from config import *

def first_fit(network_service, length, env):

    env.clear()
    env.network_service = network_service
    placement = []
    env.first_slots = -np.ones(length, dtype='int32')
    current_cpu = 0


    for i in range(length):

        vnf = network_service[i]
        size = env.vnfd_properties[vnf]["size"]

        list_cpus = list(range(env.num_cpus))[current_cpu:] + list(range(env.num_cpus))[:current_cpu]

        for cpu in list_cpus:

            change_cpu = True if placement != [] and placement[i-1] != current_cpu else False

            # Check if sufficient space in the server and in case of changing server verify also the network available
            if size <= (env.cpu_properties[cpu]["numSlots"] - env.cpu_used[cpu]) and (2*env.bandwidth <= (env.link_properties[cpu]["bandwidth"] - env.link_used[cpu]) if change_cpu else True):

                # Locate VNF
                for slot in range(size):
                    occupied_slot = env._placeSlot(cpu, vnf)

                    # Anotate first slot used by the VNF
                    if slot == 0:
                        env.first_slots[i] = occupied_slot

                # Collect placement vector
                placement.append(cpu)

                # Update CPU usage
                env.cpu_used[cpu] += env.vnfd_properties[vnf]["size"]

                # Update BW usage
                env.service_length = i+1
                env.link_used = np.zeros(env.num_cpus)

                env.placement = placement
                env._computeLink()

                break

            else:
                current_cpu = cpu + 1

    if len(placement) != length:
        return None, 0, 0, 0, 0


    env.placement = placement

    env.link_used = np.zeros(env.num_cpus)
    env.link_latency = 0
    env._computeLink()
    env._computeReward()

    return placement, env.reward, env.constraint_occupancy, env.constraint_bandwidth, env.constraint_latency

if __name__ == "__main__":

    """ Configuration """
    config, _ = get_config()

    """ Environment """
    env = Environment(config.num_cpus, config.num_vnfd, config.env_profile)

    network_service = [6, 2, 3, 3, 3, 3 ,6 ,6, 2, 5, 1, 3]
    servivce_length = 12

    placement = first_fit(network_service, servivce_length, env)
