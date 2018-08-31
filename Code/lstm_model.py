import tensorflow as tf

vocab_size = 10000
embedding_size = 512
lstm_size = 512
num_layers = 2
rnn_mode = "basic"
batch_size = 64
num_steps = 19  #auto-set?

# word embeddings matrix and lookup
with tf.device("/cpu:0"):
    embedding = tf.get_variable("embedding", [vocab_size, embedding_size], dtype=tf.float32)
    # input_.input_data shape = batch_size*num_steps
    # inputs shape = batch_size*num_steps*embedding_size
    inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
    output, state = _build_rnn_graph_lstm(inputs, config, is_training)

def _get_lstm_cell(is_training):
    if rnn_mode == "basic":
      return tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=0.0, state_is_tuple=True, reuse=not is_training)
    if rnn_mode == "block":
      return tf.contrib.rnn.LSTMBlockCell(lstm_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

def make_cell(is_training):
  cell = _get_lstm_cell(is_training)
  # if is_training and config.keep_prob < 1:
  #   cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
  return cell

def _build_rnn_graph_lstm(inputs, config, is_training):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(num_layers)], state_is_tuple=True)
    _initial_state = cell.zero_state(batch_size, tf.float32)
    #   _initial_state = (lstm_state_tuple,lstm_state_tuple_1)
    #   lstm_state_tuple shape = batch_size*embedding_size
    state = _initial_state
    # Simplified version of tf.nn.static_rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    # outputs, state = tf.nn.static_rnn(cell, inputs,
    #                                   initial_state=self._initial_state)
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    output = tf.reshape(tf.concat(outputs, 1), [-1, lstm_size])
    return output, state

