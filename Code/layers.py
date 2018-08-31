import tensorflow as tf
import numpy as np
from utils import _p, norm_weight, ortho_weight, tanh, linear, batch_matmul

class Layers(object):

    def __init__(self):
        # layers: 'name': ('parameter initializer', 'feedforward')
        self.layers = {
            'ff': ('self.param_init_fflayer', 'self.fflayer'),
            'lstm': ('self.param_init_lstm', 'self.param_init_lstm'),
            'lstm_cond': ('self.param_init_lstm_cond', 'self.lstm_cond_layer'),
            }

    def get_layer(self, name):
        """
        Part of the reason the init is very slow is because,
        the layer's constructor is called even when it isn't needed
        """
        fns = self.layers[name]
        return (eval(fns[0]), eval(fns[1]))

    def param_init_fflayer(self, options, params, prefix='ff', nin=None, nout=None):
        if nin == None:
            nin = options['ctx_dim']
        if nout == None:
            nout = options['lstm_dim']
        params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01)
        # tfparams[_p(prefix, 'W')] = tf.placeholder(tf.float32, shape=(nin,nout), name=_p(prefix, 'W'))
        params[_p(prefix, 'b')] = np.zeros((nout,)).astype('float32')
        # tfparams[_p(prefix, 'b')] = tf.placeholder(tf.float32, shape=(nout,), name=_p(prefix, 'b'))
        return params

    def fflayer(self, tfparams, state_below, options, 
                prefix='rconv', activ='lambda x: tf.tanh(x)', **kwargs):
        return eval(activ)(tf.matmul(state_below, tfparams[_p(prefix, 'W')]) + tfparams[_p(prefix, 'b')])

    # LSTM layer
    def param_init_lstm(self, params, nin, dim, prefix='lstm'):
        assert prefix is not None
        # Stack the weight matricies for faster dot prods
        W = np.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[_p(prefix, 'W')] = W     # to_lstm_W:(512,2048)
        # tfparams[_p(prefix, 'W')] = tf.placeholder(tf.float32, shape=(W.shape[0],W.shape[1]), name=_p(prefix, 'W'))
        U = np.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix, 'U')] = U     # to_lstm_U:(512,2048)
        # tfparams[_p(prefix, 'U')] = tf.placeholder(tf.float32, shape=(U.shape[0],U.shape[1]), name=_p(prefix, 'U'))
        params[_p(prefix, 'b')] = np.zeros((4 * dim,)).astype('float32')    # to_lstm_b:(2048,)
        # tfparams[_p(prefix, 'b')] = tf.placeholder(tf.float32, shape=(4 * dim,), name=_p(prefix, 'b'))
        return params

    # Conditional LSTM layer with Attention
    def param_init_lstm_cond(self, options, params,
                             prefix='lstm_cond', nin=None, dim=None, dimctx=None):  #nin=512 dim=512 dimctx=2048
        if nin == None:
            nin = options['word_dim']
        if dim == None:
            dim = options['lstm_dim']
        if dimctx == None:
            dimctx = options['ctx_dim']
        # input to LSTM
        W = np.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[_p(prefix, 'W')] = W     # bo_lstm_W:(512,2048)
        # tfparams[_p(prefix, 'W')] = tf.placeholder(tf.float32, shape=(W.shape[0],W.shape[1]), name=_p(prefix, 'W'))
        # LSTM to LSTM
        U = np.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)], axis=1)
        params[_p(prefix, 'U')] = U     # bo_lstm_U:(512,2048)
        # tfparams[_p(prefix, 'U')] = tf.placeholder(tf.float32, shape=(U.shape[0],U.shape[1]), name=_p(prefix, 'U'))
        # bias to LSTM
        params[_p(prefix, 'b')] = np.zeros((4 * dim,)).astype('float32')      # bo_lstm_b:(2048,)
        # tfparams[_p(prefix, 'b')] = tf.placeholder(tf.float32, shape=(4 * dim,), name=_p(prefix, 'b'))
        # attention: context -> hidden
        Wc_att = norm_weight(dimctx, ortho=False)
        params[_p(prefix, 'Wc_att')] = Wc_att    # bo_lstm_Wc_att:(2048,2048)
        # tfparams[_p(prefix, 'Wc_att')] = tf.placeholder(tf.float32, shape=(Wc_att.shape[0],Wc_att.shape[1]), name=_p(prefix, 'Wc_att'))
        # attention: LSTM -> hidden
        Wd_att = norm_weight(dim, dimctx)
        params[_p(prefix, 'Wd_att')] = Wd_att   # bo_lstm_Wd_att:(512,2048)
        # tfparams[_p(prefix, 'Wd_att')] = tf.placeholder(tf.float32, shape=(Wd_att.shape[0],Wd_att.shape[1]), name=_p(prefix, 'Wd_att'))
        # attention: hidden bias
        b_att = np.zeros((dimctx,)).astype('float32')
        params[_p(prefix, 'b_att')] = b_att     # bo_lstm_b_att:(2048,)
        # tfparams[_p(prefix, 'b_att')] = tf.placeholder(tf.float32, shape=(dimctx,), name=_p(prefix, 'b_att'))
        # attention:
        U_att = norm_weight(dimctx, 1)
        params[_p(prefix, 'U_att')] = U_att      # bo_lstm_U_att:(2048,1)
        # tfparams[_p(prefix, 'U_att')] = tf.placeholder(tf.float32, shape=(U_att.shape[0],U_att.shape[1]), name=_p(prefix, 'U_att'))
        c_att = np.zeros((1,)).astype('float32')
        params[_p(prefix, 'c_att')] = c_att  # bo_lstm_c_att:(1,)
        # tfparams[_p(prefix, 'c_att')] = tf.placeholder(tf.float32, shape=(1,), name=_p(prefix, 'c_att'))
        if options['selector']:
            # attention: selector
            W_sel = norm_weight(dim, 1)
            params[_p(prefix, 'W_sel')] = W_sel     # bo_lstm_W_sel:(512,1)
            # tfparams[_p(prefix, 'W_sel')] = tf.placeholder(tf.float32, shape=(W_sel.shape[0],W_sel.shape[1]), name=_p(prefix, 'W_sel'))
            b_sel = np.float32(0.)
            params[_p(prefix, 'b_sel')] = b_sel     # bo_lstm_b_sel: 0
            # tfparams[_p(prefix, 'b_sel')] = tf.placeholder(tf.float32, name=_p(prefix, 'b_sel'))
        return params

    def lstm_cond_layer(self, tfparams, state_below, options, prefix,
                        mask, context, one_step,
                        init_memory, init_state,
                        use_noise,
                        **kwargs):
        # state_below (t, m, dim_word), or (m, dim_word) in sampling
        # mask (t, m)
        # context (m, f, dim_ctx), or (f, dim_word) in sampling
        # init_memory, init_state (m, dim)
        # t = time steps
        # m = batch size
        nsteps = state_below.shape[0]
        if state_below.shape.ndims == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1
        dim = tfparams[_p(prefix, 'U')].shape[0]
        # projected context
        pctx_ = batch_matmul(context, tfparams[_p(prefix, 'Wc_att')]) + tfparams[_p(prefix, 'b_att')]    # (64,28,2048)*(2048,2048)+(2048,) = (64,28,2048)
        if one_step:
            # tensor.dot will remove broadcasting dim
            # pctx_ = tensor.addbroadcast(pctx_, 0)
            raise NotImplementedError()
        # projected x
        state_below = batch_matmul(state_below, tfparams[_p(prefix, 'W')]) + tfparams[_p(prefix, 'b')]    # (19,64,512)*(512,2048)+(2048) = (19,64,2048)
        Wd_att = tfparams[_p(prefix, 'Wd_att')]  # (512,2048)
        U_att = tfparams[_p(prefix, 'U_att')]    # (2048,1)
        c_att = tfparams[_p(prefix, 'c_att')] # (1,)
        if options['selector']:
            W_sel = tfparams[_p(prefix, 'W_sel')]
            b_sel = tfparams[_p(prefix, 'b_sel')]
        else:
            # W_sel = tensor.alloc(0., 1)
            # b_sel = tensor.alloc(0., 1)
            raise NotImplementedError()
        U = tfparams[_p(prefix, 'U')]    # (512,2048)

        def _slice(_x, n, dim):
            if _x.shape.ndims == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def step(prev, elems):
            # gather previous internal state and output state
            m_, x_ = elems
            h_, c_ = tf.unstack(prev)
            preact = tf.matmul(h_, U)   # (64,512)*(512,2048) = (64,2048)
            preact += x_
            i = _slice(preact, 0, dim)  # (64,512)  (0-511)
            f = _slice(preact, 1, dim)  # (64,512)  (512,1023)
            o = _slice(preact, 2, dim)  # (64,512)  (1024-1535)
            if options['use_dropout']:
                raise NotImplementedError()
            i = tf.sigmoid(i)
            f = tf.sigmoid(f)
            o = tf.sigmoid(o)
            c = tf.tanh(_slice(preact, 3, dim))  # (64,512)  (1024-1535)
            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_
            h = o * tf.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_
            return tf.stack([h, c])

        states = tf.scan(step, 
                    (mask,state_below),
                    initializer=tf.stack([init_state,init_memory]),
                    name=_p(prefix, '_layers'))

        return states

    def create_get_lstm_cell(prefix, is_training=True, rnn_mode="basic"):
        if rnn_mode == "basic":
          return tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=0.0, state_is_tuple=True, reuse=not is_training, name=prefix)
        if rnn_mode == "block":
          return tf.contrib.rnn.LSTMBlockCell(lstm_size, forget_bias=0.0, name=prefix)
        raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

    def make_cell(prefix, is_training=True, rnn_mode="basic"):
        cell = _get_lstm_cell(is_training, rnn_mode)
        # if is_training and config.keep_prob < 1:
        #   cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
        return cell

    # Incomplete
    def build_hlstm(prefix, inputs, n_timesteps, init_state, init_memory, is_training=True, rnn_mode="basic"):
        """Build the inference graph using canonical LSTM cells."""
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        cell = tf.contrib.rnn.MultiRNNCell([make_cell(lstm_name, is_training, rnn_mode) for lstm_name in prefix], state_is_tuple=True)
        # Simplified version of tf.nn.static_rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        # outputs, state = tf.nn.static_rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        with tf.variable_scope("RNN"):
          for time_step in range(n_timesteps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[time_step, :, :], state)
            outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, lstm_size])
        return output, state