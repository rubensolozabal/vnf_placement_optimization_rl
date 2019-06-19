import numpy as np


class ServiceBatchGenerator(object):
    """
        Implementation of a random service chain generator

        Attributes:
            state[batch_size, max_service_length] -- Batch of random service chains
            service_length[batch_size]            -- Array containing services length
    """
    def __init__(self, batch_size, min_service_length, max_service_length, vocab_size):
        """
        Args:
            batch_size(int)         -- Number of service chains to be generated
            min_service_length(int) -- Minimum service length
            max_service_length(int) -- Maximum service length
            vocab_size(int)         -- Size of the VNF dictionary
        """
        self.batch_size = batch_size
        self.min_service_length = min_service_length
        self.max_service_length = max_service_length
        self.vocab_size = vocab_size

        self.service_length = np.zeros(self.batch_size,  dtype='int32')
        self.state = np.zeros((self.batch_size, self.max_service_length),  dtype='int32')

    def getNewState(self):
        """ Generate new batch of service chain """

        # Clean attributes
        self.state = np.zeros((self.batch_size, self.max_service_length), dtype='int32')
        self.service_length = np.zeros(self.batch_size,  dtype='int32')

        # Compute random services
        for batch in range(self.batch_size):
            self.service_length[batch] = np.random.randint(self.min_service_length, self.max_service_length+1, dtype='int32')
            for i in range(self.service_length[batch]):
                vnf_id = np.random.randint(1, self.vocab_size,  dtype='int32')
                self.state[batch][i] = vnf_id


if __name__ == "__main__":

    # Define generator
    batch_size = 5
    min_service_length = 2
    max_service_length = 6
    vocab_size = 8

    env = ServiceBatchGenerator(batch_size, min_service_length, max_service_length, vocab_size)
    env.getNewState()


