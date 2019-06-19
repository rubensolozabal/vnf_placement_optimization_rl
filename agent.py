# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt

def variable_summaries(name, var, with_max_min=False):
    """ Tensor summaries for TensorBoard visualization """

    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)

        if with_max_min == True:
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))


class Seq2seqModel:
    """
        Implementation of a sequence-to-sequence model based on dynamic multi-cell RNNs

        Attributes:
            batch_size(int)                         -- Batch of random service chains
            action_size(int)                        -- Number of hosts
            embeddings(int)                         -- Embedding size
            length(int)                             -- Maximum sequence size
            hidden_size(int)                        -- LSTM hidden size
            num_layers(int)                         -- No stacked LSTM cells
    """

    def __init__(self, config, input_, input_len_, mask):

        self.action_size = config.num_cpus
        self.batch_size = config.batch_size
        self.embeddings = config.embedding_size
        self.state_size = config.num_vnfd
        self.length = config.max_length
        vocab_size = config.num_vnfd + 1

        self.hidden_size = config.hidden_dim
        self.num_layers = config.num_layers

        self.positions = []
        self.outputs = []

        self.input_ = input_
        self.input_len_ = input_len_
        self.mask = mask

        self.initialization_stddev = 0.1
        self.attention_plot = []

        with tf.variable_scope("actor"):

            # Define encoder block
            with tf.variable_scope("actor_encoder"):

                # Variables initializer
                initializer = tf.contrib.layers.xavier_initializer()

                # Embeddings
                embeddings = tf.Variable(tf.random_uniform([vocab_size, self.embeddings], -1.0, 1.0),
                                         dtype=tf.float32)

                embedded_input = tf.nn.embedding_lookup(embeddings, input_)

                # Generate multiple LSTM cell
                enc_cells = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True) for _ in range(self.num_layers)],
                    state_is_tuple=True)

                c_initial_states = []
                h_initial_states = []

                # Initial state (tuple) is trainable but same for all batch
                for i in range(self.num_layers):
                    first_state = tf.get_variable("var{}".format(i), [1, self.hidden_size], initializer=initializer)
                    # first_state = tf.Print(first_state, ["first_state", first_state], summarize=10)

                    c_initial_state = tf.tile(first_state, [self.batch_size, 1])
                    h_initial_state = tf.tile(first_state, [self.batch_size, 1])

                    c_initial_states.append(c_initial_state)
                    h_initial_states.append(h_initial_state)


                # LSTM stack
                rnn_tuple_state = tuple(
                    [tf.nn.rnn_cell.LSTMStateTuple(c_initial_states[idx], h_initial_states[idx])
                     for idx in range(self.num_layers)])

                # LSTM output
                self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(cell=enc_cells, inputs=embedded_input, initial_state=rnn_tuple_state, dtype=tf.float32)

                #enc_outputs, enc_final_state = tf.nn.dynamic_rnn(cell=enc_cells, inputs=embedded_input, initial_state=rnn_tuple_state,
                #                                                 sequence_length=input_len_, dtype=tf.float32)

            # Define decoder block
            with tf.variable_scope("actor_decoder"):

                # LSTM stack
                decoder_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=True) for _ in range(self.num_layers)],
                    state_is_tuple=True)

                first_process_block_input = tf.tile(tf.Variable(tf.random_normal([1, self.hidden_size]),
                                                                name='first_process_block_input'), [self.batch_size, 1])

                # Define attention weights
                with tf.variable_scope("actor_attention_weights", reuse=True):

                    W_ref = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size], stddev=self.initialization_stddev), name='W_ref')
                    W_q = tf.Variable(tf.random_normal([self.hidden_size, self.hidden_size], stddev=self.initialization_stddev), name='W_q')
                    v = tf.Variable(tf.random_normal([self.hidden_size], stddev=self.initialization_stddev), name='v')


                # Processing chain
                decoder_state = self.encoder_final_state
                decoder_input = tf.unstack(self.encoder_outputs, num=None, axis=1, name='unstack') #first_process_block_input

                decoder_outputs = []
                decoder_attLogits = []

                for t in range(self.length):
                    decoder_output, decoder_state = decoder_cell(inputs=decoder_input[t], state=decoder_state)

                    #dec_output = tf.layers.dense(dec_output, self.embedding, tf.nn.relu)
                    decoder_outputs.append(decoder_output)
                    #decoder_input = decoder_output
                    #_, attnLogits, context = self.attention(W_ref, W_q, v, attnInputs=enc_outputs, query=dec_output, mask=self.mask)
                    #dec_attLogits.append(attnLogits)
                    #dec_input = context

                dec_outputs = tf.stack(decoder_outputs, axis=1)
                #self.attention_plot = tf.stack(dec_attLogits, axis=1)

            self.decoder_logits = tf.layers.dense(dec_outputs, self.action_size)         # [Batch, seq_length, action_size]

            # Multinomial distribution
            self.decoder_softmax = tf.nn.softmax(self.decoder_logits)
            prob = tf.contrib.distributions.Categorical(probs=self.decoder_softmax)

            # Sample from distribution
            self.decoder_exploration = prob.sample(1)
            self.decoder_exploration = tf.cast(self.decoder_exploration, tf.int32)

            # Decoder prediction
            self.decoder_prediction = tf.argmax(self.decoder_logits, 2)
            self.decoder_prediction = tf.expand_dims(self.decoder_prediction, 0)

            # Sampling
            temperature = 15
            self.decoder_softmax_temp = tf.nn.softmax(self.decoder_logits / temperature)

            prob = tf.contrib.distributions.Categorical(probs=self.decoder_softmax)

            self.samples = 16
            self.decoder_sampling = prob.sample(self.samples)

    def attention(self, W_ref, W_q, v, attnInputs, query, mask=None, maskPenalty = 10^6):
        """
        Attention mechanism in Vinyals (2015)
        attnInputs are the states over which to attend over
        """

        with tf.variable_scope("RNN_Attention"):
            u_i0s = tf.einsum('kl,itl->itk', W_ref, attnInputs)
            u_i1s = tf.expand_dims(tf.einsum('kl,il->ik', W_q, query), 1)
            unscaledAttnLogits = tf.einsum('k,itk->it', v, tf.tanh(u_i0s + u_i1s))
            #unscaledAttnLogits = tf.Print(unscaledAttnLogits, ["unscaledAttnLogits", unscaledAttnLogits, tf.shape(unscaledAttnLogits)], summarize=10)

            if mask is not None:
                maskedUnscaledAttnLogits = unscaledAttnLogits - tf.multiply(mask, maskPenalty)
                #maskedUnscaledAttnLogits = tf.Print(maskedUnscaledAttnLogits, ["maskedUnscaledAttnLogits", maskedUnscaledAttnLogits, tf.shape(maskedUnscaledAttnLogits)], summarize=10)

            attnLogits = tf.nn.softmax(maskedUnscaledAttnLogits)
            #attnLogits = tf.Print(attnLogits,["attnLogits", attnLogits, tf.shape(attnLogits)], summarize=10)

            context = tf.einsum('bi,bic->bc', attnLogits, attnInputs)

        return unscaledAttnLogits, attnLogits, context

    def plot_attention(self, attention):
        """ Plot the attention """

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention, cmap='viridis')

        fontdict = {'fontsize': 14}

        ax.set_xticklabels(['sentence'], fontdict=fontdict, rotation=90)
        ax.set_yticklabels(['predicted'], fontdict=fontdict)

        plt.show()


class ValueEstimator():
    """
        Value Function approximator

        Attributes:
            batch_size(int)                         -- Batch of random service chains
            embeddings(int)                         -- Embedding size
            length(int)                             -- Maximum sequence size
            hidden_size(int)                        -- LSTM hidden size
    """

    def __init__(self, config, input_):

        with tf.variable_scope("value_estimator"):

            self.embeddings = config.embedding_size
            self.length = config.max_length
            vocab_size = config.num_vnfd + 1

            #self.state = tf.placeholder(tf.int32, [], "state")
            self.target = tf.placeholder(tf.float32, [config.batch_size], name="target")

            # Embeddings
            embeddings = tf.Variable(tf.random_uniform([vocab_size, self.embeddings], -1.0, 1.0),
                                     dtype=tf.float32)

            embedded_input = tf.nn.embedding_lookup(embeddings, input_)

            # Encoder
            encoder_cell = tf.contrib.rnn.LSTMCell(config.hidden_dim)

            _, encoder_final_state = tf.nn.dynamic_rnn(
                encoder_cell, embedded_input,dtype=tf.float32)

            # MLP output layer
            output = tf.layers.dense(encoder_final_state.h, 1)

            self.value_estimate = tf.squeeze(output)
            target = self.target

            self.loss = tf.squared_difference(self.value_estimate, target)
            #variable_summaries('valueEstimator_loss', self.loss, with_max_min=False)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())


class Agent:
    """
        Agent composed by a sequence-to-sequence model and a baseline estimator
    """
    def __init__(self, config):

        # Training config (agent)
        self.learning_rate = config.learning_rate
        #self.global_step = tf.Variable(0, trainable=False, name="global_step")  # global step
        #self.lr_start = config.lr1_start  # initial learning rate
        #self.lr_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        #self.lr_decay_step = config.lr1_decay_step  # learning rate decay step

        self.action_size = config.num_cpus
        self.batch_size = config.batch_size
        self.embeddings = config.embedding_size
        self.state_size = config.num_vnfd
        self.length = config.max_length

        self.lambda_occupancy = 1000
        self.lambda_bandwidth = 10
        self.lambda_latency = 10

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = tf.placeholder(tf.int32, [self.batch_size, self.length], name="input")
        self.input_len_ = tf.placeholder(tf.int32, [self.batch_size], name="input_len")
        self.mask = tf.placeholder(tf.int32, [self.batch_size, self.length], name="mask")

        self._build_model(config)
        self._build_ValueEstimator(config)
        self._build_optimization()

        self.merged = tf.summary.merge_all()

    def _build_model(self, config):

        with tf.variable_scope('actor'):

            self.actor = Seq2seqModel(config,  self.input_, self.input_len_, self.mask)

    def _build_ValueEstimator(self, config):

        with tf.variable_scope('value_estimator'):

            self.valueEstimator = ValueEstimator(config, self.input_)

    def _build_optimization(self):

        with tf.name_scope('reinforce_learning'):

            self.placement_holder = tf.placeholder(tf.float32, [self.batch_size, self.length], name="placement_holder")
            self.baseline_holder = tf.placeholder(tf.float32, [self.batch_size], name="baseline_holder")
            self.lagrangian_holder = tf.placeholder(tf.float32, [self.batch_size], name="lagrangian_holder")

            # Optimizer learning rate
            #self.opt = tf.train.exponential_decay(self.lr1_start, self.global_step, self.lr1_decay_step, self.lr1_decay_rate, staircase=False, name="learning_rate1")
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99, epsilon=0.0000001)

            # Multinomial distribution
            probs = tf.contrib.distributions.Categorical(probs=self.actor.decoder_softmax)
            log_softmax = probs.log_prob(self.placement_holder)         # [Batch, seq_length]
            # log_softmax = tf.Print(log_softmax, ["log_softmax", log_softmax, tf.shape(log_softmax)])

            log_softmax_mean =  tf.reduce_mean(log_softmax,1)                  # [Batch]
            # log_softmax_mean = tf.Print(log_softmax_mean, ["log_softmax_mean",log_softmax_mean, tf.shape(log_softmax_mean)])
            variable_summaries('log_softmax_mean', log_softmax_mean, with_max_min=True)

            self.advantage = self.lagrangian_holder - self.baseline_holder
            variable_summaries('adventage', self.advantage, with_max_min=False)

            # Compute Loss
            self.loss_rl = tf.reduce_mean(self.advantage * log_softmax_mean, 0)     # Scalar

            tf.summary.scalar('loss', self.loss_rl)

            # Minimize step
            gvs = opt.compute_gradients(self.loss_rl)

            # Clipping
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None]  # L2 clip

            self.train_step = opt.apply_gradients(capped_gvs)


if __name__ == "__main__":

    pass
