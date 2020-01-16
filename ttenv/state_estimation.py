import tensorflow as tf
import random
import numpy as np
import copy
import models
import tensorflow.contrib.layers as layers

BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"

def data_type():
    return tf.float32

def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

class BeliefTargetModel(object):
    """A Dynamic Belief Target Model."""

    def __init__(self,
                is_training = True,
                learning_rate = 0.01,
                max_grad_norm = 5,
                num_layers = 2,
                num_steps = 20,
                hidden_size = 64,
                keep_prob = 1.0,
                lr_decay = 0.999,
                batch_size = 32,
                dim=(None, None),
                rnn_mode = BLOCK,
                output_b_init=None):
        """
        is_training : True if there is a new observation data. False if you want to just predict
        """
        self._is_training = is_training
        self._rnn_params = None
        self._cell = None
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.rnn_mode = rnn_mode
        self.keep_prob = keep_prob
        self.inputs = tf.placeholder(tf.float32,
                        shape=[batch_size, num_steps, dim[0]],
                        name="inputs")
        self.targets = tf.placeholder(tf.float32,
                        shape=[batch_size, num_steps, dim[1]],
                        name="targets")

        if is_training and keep_prob < 1:
            self.inputs = tf.nn.dropout(self.inputs, keep_prob)

        output, state = self._build_rnn_graph_lstm(self.inputs, is_training)
        output_w = tf.get_variable("output_w", [hidden_size, dim[1]], dtype=data_type())
        if output_b_init is not None:
            output_b = tf.get_variable("output_b", [dim[1]], dtype=data_type(),
                                initializer=tf.random_normal_initializer(mean=output_b_init, stddev=0.01))
        else:
            output_b = tf.get_variable("output_b", [dim[1]], dtype=data_type())
        self.outputs = tf.nn.xw_plus_b(output, output_w, output_b)
        self.outputs = tf.reshape(self.outputs, [self.batch_size, self.num_steps, dim[1]])
        # Unlike the language model, the output is continuous and there is no softmax layer.
        # loss = tf.reduce_mean(huber_loss(tf.slice(self.outputs - self.targets, [0, 1, 0], [-1, -1, -1])), axis=1)
        err = self.outputs - self.targets
        loss = tf.reduce_mean(tf.square(err), axis=-1)

        err = tf.reshape(err, [-1, dim[1]])
        self.err_cov = tf.divide(tf.matmul(err,err, transpose_a=True), batch_size)
        #loss = tf.reduce_mean(huber_loss(self.outputs - self.targets), axis=1)

        # Update the cost
        self._cost = tf.reduce_mean(loss) # Across the batch
        self._final_state = state
        self.initial_state_val = None

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), max_grad_norm)
        optimizer = tf.train.RMSPropOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _get_lstm_cell(self, is_training):
        if self.rnn_mode == BASIC:
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0,
                                        state_is_tuple=True, reuse=not is_training)
        if self.rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(self.hidden_size, forget_bias=1.0)
        raise ValueError("rnn_mode %s not supported"%self.rnn_mode)

    def assign_lr(self, session, lr_value=None):
        if lr_value is None:
            lr_value = self.learning_rate
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def _build_rnn_graph_lstm(self, inputs, is_training):
        """Build the inference grpah using canonical LSTM cells."""
        def make_cell():
            cell = self._get_lstm_cell(is_training)
            if is_training and self.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                        cell, output_keep_prob=self.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
                [make_cell() for _ in range(self.num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(self.batch_size, data_type())
        state = self._initial_state

        inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        outputs, state = tf.nn.static_rnn(cell, inputs, initial_state=self._initial_state)
        output = tf.reshape(tf.concat(outputs,1), [-1, self.hidden_size])
        return output, state

    def update(self, session, inputs, targets):
        #state = session.run(self.initial_state) #?
        state = self.initial_state_val if self.initial_state_val else session.run(self.initial_state)
        fetches = {"cost": self.cost, "final_state": self.final_state, "err_cov": self.err_cov}#, "outputs":self.outputs}
        if self.train_op is not None:
            fetches["train_op"] = self.train_op
        feed_dict = {self.inputs : inputs, self.targets : targets}
        for i, (c,h) in enumerate(self.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        vals = session.run(fetches, feed_dict)
        self.initial_state_val = vals["final_state"]
        return vals["cost"], vals["err_cov"]

    def predict(self, session, inputs):
        state = self.initial_state_val if self.initial_state_val else session.run(self.initial_state)
        fetches = {"output": self.outputs, "final_state": self.final_state}
        feed_dict = {self.inputs : inputs}
        for i, (c,h) in enumerate(self.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        vals = session.run(fetches, feed_dict)
        self.initial_state_val = vals["final_state"]
        return vals["output"][0][0]

    def reset(self, lr_decay=False, session=None):
        self.initial_state_val = None
        if lr_decay:
            self.assign_lr(session, session.run(self._lr) * self.lr_decay)

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name

class BeliefTargetModelMLP(object):
    def __init__(self,
                is_training = True,
                learning_rate = 0.001,
                max_grad_norm = 5,
                num_layers = 2,
                hidden_size = 64,
                keep_prob = 1.0,
                lr_decay = 0.999,
                batch_size = 32,
                dim = (None, None)):
        self.inputs = tf.placeholder(tf.float32, shape=[batch_size, dim[0]], name="inputs")
        self.targets = tf.placeholder(tf.float32, shape=[batch_size, dim[1]], name="targets")
        out = self.input
        for _ in range(num_layers):
            hidden = hidden_size
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)
        self.outputs = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        self._cost = tf.reduce_mean(tf.huber_loss(self.outputs - self.targets))

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), max_grad_norm)
        optimizer = tf.train.RMSPropOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

class TargetHistory(object):
    def __init__(self, memory_size, num_steps, batch_size):
        self._memory_size = memory_size
        self._memory = []
        self.batch_size = batch_size
        self.num_steps = num_steps
        self._current_rnn_inputs = []
        self._current_rnn_targets = []

    def get_batch(self):
        samples_i = []
        samples_t = []
        for _ in range(self.batch_size):
            rnn_inputs, rnn_targets = random.choice(self._memory)
            samples_i.append(rnn_inputs)
            samples_t.append(rnn_targets)
        return samples_i, samples_t

    def store(self, rnn_input, rnn_target):
        self._current_rnn_inputs.append(rnn_input)
        self._current_rnn_targets.append(rnn_target)
        if len(self._current_rnn_inputs) == self.num_steps:
            if (self._current_rnn_inputs != np.zeros((5,))).any():
                self._memory.append((copy.deepcopy(self._current_rnn_inputs),
                                    copy.deepcopy(self._current_rnn_targets)))
            self._current_rnn_inputs.pop(0)
            self._current_rnn_targets.pop(0)

        if len(self._memory) == self._memory_size:
            self._memory.pop(0)

    def reset(self):
        self._current_rnn_inputs = []
        self._current_rnn_targets = []

    def __len__(self):
        return len(self._memory)


"""To run
    import state_estimation *
    target_model = TargetModel(True, config)
    batch_data =
    target_model.update(inputs, targets)
"""
