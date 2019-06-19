import pymzn
import pymzn.config
from service_batch_generator import *
from environment import *
from config import *


pymzn.debug(False)


def solver(network_service, length, env):

    chain = network_service

    if not isinstance(network_service, (list,)):
        chain = network_service.tolist()

    chain = chain[:length]

    slots = [env.cpu_properties[i]["numSlots"] for i in range(env.num_cpus)]
    link_bandwidth = [env.link_properties[i]["bandwidth"] for i in range(env.num_cpus)]
    link_latency = [env.link_properties[i]["latency"] for i in range(env.num_cpus)]

    vnf_weights = [env.vnfd_properties[i]['size'] for i in range(1, env.num_vnfds + 1)]
    vnf_bandwidth = [env.vnfd_properties[i]['bandwidth'] for i in range(1, env.num_vnfds + 1)]
    vnf_latency = [env.vnfd_properties[i]['latency'] for i in range(1, env.num_vnfds + 1)]

    placement = service_bandwidth = service_net_latency = service_cpu_latency = energy = occupancy = link_used = None

    s = pymzn.minizinc('placement.mzn', timeout=30, parallel=4, data={'numServers': env.num_cpus,
                                                                        'maxSlots': env.max_slots,
                                                                        'slots': slots,
                                                                        'link_bandwidth': link_bandwidth,
                                                                        'link_latency': link_latency,

                                                                        'chainLen': length,
                                                                        'chain': chain,

                                                                        'numVnfds': env.num_vnfds,
                                                                        'vnf_weights': vnf_weights,
                                                                        'vnf_bandwidth': vnf_bandwidth,
                                                                        'vnf_latency': vnf_latency,
                                                                        'p_min': env.p_min,
                                                                        'p_slot': env.p_slot
                                                                     })

    try:
        placement = s[0]['placement']
        placement = [x - 1 for x in placement]
        service_bandwidth = s[0]['service_bandwidth']
        service_net_latency = s[0]['service_net_latency']
        service_cpu_latency = s[0]['service_cpu_latency']
        energy = s[0]['energy']
        occupancy = s[0]['occupancy']
        link_used = s[0]['link_used']

    except:
        print("Solution not found!")

    return placement, service_bandwidth, service_net_latency, service_cpu_latency, energy, occupancy, link_used


if __name__ == "__main__":
    """ Configuration """
    config, _ = get_config()

    """ Environment """
    env = Environment(config.num_cpus, config.num_vnfd)

    """ Network service generator """
    vocab_size = config.num_vnfd + 1
    network_services = ServiceBatchGenerator(1, config.min_length, config.max_length, vocab_size)

    # New batch of states
    network_services.getNewState()

    print("Computing... ")

    ns = [2, 6, 3, 5, 5, 3, 7, 4, 1, 3, 5, 4]
    len = 12
    optPlacement, service_bandwidth, service_net_latency, service_cpu_latency, energy, occupancy, link_used = solver(ns, len, env)

    print("Optimal placement: ", optPlacement)
    print("Service bandwidth: ", service_bandwidth)
    print("Service network latency: ", service_net_latency)
    print("Service CPU latency: ", service_cpu_latency)
    print("Energy: ", energy)
    print("Occupancy: ", occupancy)
    print("Link bandwidth used: ", link_used)
