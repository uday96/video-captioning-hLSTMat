import argparse
import logging
import utils, config
import time
from model import Model
import data_engine
import tensorflow as tf
import numpy as np

def train(model_options,
        dataset_name = 'msvd',
        cnn_name = 'resnet',
        train_data_ids_path = config.MSVD_DATA_IDS_TRAIN_PATH,
        val_data_ids_path = config.MSVD_DATA_IDS_VAL_PATH,
        test_data_ids_path = config.MSVD_DATA_IDS_TEST_PATH,
        vocab_path = config.MSVD_VOCAB_PATH,
        reverse_vocab_path = config.MSVD_REVERSE_VOCAB_PATH,
        mb_size_train = 64,
        mb_size_test = 128,
        train_caps_path = config.MSVD_VID_CAPS_TRAIN_PATH,
        val_caps_path = config.MSVD_VID_CAPS_VAL_PATH,
        test_caps_path = config.MSVD_VID_CAPS_TEST_PATH,
        feats_dir = config.MSVD_FEATS_RESNET_DIR,
        save_dir = config.SAVE_DIR_PATH,
        word_dim = 512,   # word embeddings size
        ctx_dim = 2048,   # video cnn feature dimension
        lstm_dim = 512,   # lstm unit size
        patience = 20,
        max_epochs = 500,
        lrate = 0.0001,
        vocab_size = 20000, # n_words
        maxlen_caption = 30,  # max length of the descprition
        optimizer = 'adadelta',
        batch_size = 64,  # for trees use 25
        metric = 'everything',    # set to perplexity on DVS # blue, meteor, or both
        use_dropout = True,
        selector = True,
        ctx2out = True,
        prev2out = True,
        dispFreq = 10,
        validFreq = 10,
        saveFreq = 10, # save the parameters after every saveFreq updates
        sampleFreq = 10, # generate some samples after every sampleFreq updates
        verbose = True,
        debug = False,
        reload = False,
        from_dir = '',
        ctx_frames = 28, # 26 when compare
        random_seed = 1234
        ):

    tf.set_random_seed(random_seed)

    model = Model()

    print 'Loading data'
    engine = data_engine.Movie2Caption(dataset_name,cnn_name,train_data_ids_path, val_data_ids_path, test_data_ids_path,
                vocab_path, reverse_vocab_path, mb_size_train, mb_size_test, maxlen_caption,
                train_caps_path, val_caps_path, test_caps_path, feats_dir)

    model_options['ctx_dim'] = engine.ctx_dim
    ctx_dim = engine.ctx_dim
    model_options['vocab_size'] = engine.vocab_size
    vocab_size = engine.vocab_size
    print 'n_words:', model_options['vocab_size']

    # set test values, for debugging
    idx = engine.kf_train[0]
    x_tv, mask_tv, ctx_tv, ctx_mask_tv = data_engine.prepare_data(engine, [engine.train_data_ids[index] for index in idx], mode="train")

    print 'init params'
    t0 = time.time()
    params = model.init_params(model_options)

    # description string: #words x #samples
    X = tf.placeholder(tf.int32, shape=(None, batch_size), name='word_seq_x')  # word seq input (t,m)
    MASK = tf.placeholder(tf.float32, shape=(None, batch_size), name='word_seq_mask')   # (t,m)
    # context: #samples x #annotations x dim
    CTX = tf.placeholder(tf.float32, shape=(batch_size, ctx_frames, ctx_dim), name='ctx')
    CTX_MASK = tf.placeholder(tf.float32, shape=(batch_size, ctx_frames), name='ctx_mask')

    CTX_SAMPLER = tf.placeholder(tf.float32, shape=(ctx_frames, ctx_dim), name='ctx_sampler')
    CTX_MASK_SAMPLER = tf.placeholder(tf.float32, shape=(ctx_frames), name='ctx_mask_sampler')
    X_SAMPLER = tf.placeholder(tf.int32, shape=(None,), name='x_sampler')   # DOUBT 1 or None ?
    BO_INIT_STATE_SAMPLER = tf.placeholder(tf.float32, shape=(None,lstm_dim), name='bo_init_state_sampler')
    TO_INIT_STATE_SAMPLER = tf.placeholder(tf.float32, shape=(None,lstm_dim), name='to_init_state_sampler')
    BO_INIT_MEMORY_SAMPLER = tf.placeholder(tf.float32, shape=(None,lstm_dim), name='bo_init_memory_sampler')
    TO_INIT_MEMORY_SAMPLER = tf.placeholder(tf.float32, shape=(None,lstm_dim), name='to_init_memory_sampler')

    # create tensorflow variables
    print 'buliding model'
    tfparams = utils.init_tfparams(params)
    use_noise, COST, extra = model.build_model(tfparams, model_options, X, MASK, CTX, CTX_MASK)
    ALPHAS = extra[1]
    BETAS = extra[2]

    print 'buliding sampler'
    f_init, f_next = model.build_sampler(tfparams, model_options, use_noise, CTX_SAMPLER, CTX_MASK_SAMPLER,
                                X_SAMPLER, BO_INIT_STATE_SAMPLER, TO_INIT_STATE_SAMPLER, BO_INIT_MEMORY_SAMPLER, TO_INIT_MEMORY_SAMPLER)

    print 'building f_log_probs'
    f_log_probs = -COST

    print 'building f_alpha'
    f_alpha = [ALPHAS, BETAS]

    LOSS = tf.reduce_mean(COST)
    UPDATE_OPS = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(UPDATE_OPS):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=lrate).minimize(LOSS)

    # Initialize all variables
    var_init = tf.global_variables_initializer()
    # Ops to save and restore all the variables.
    saver = tf.train.Saver()

    bad_counter = 0

    processes = None
    queue = None
    rqueue = None
    shared_params = None

    uidx = 0
    uidx_best_blue = 0
    uidx_best_valid_err = 0
    estop = False
    # best_p = utils.unzip(tparams)
    best_blue_valid = 0
    best_valid_err = 999
    alphas_ratio = []

    # Launch the graph
    with tf.Session() as sess:
        sess.run(var_init)
        for eidx in xrange(max_epochs):
            n_samples = 0
            train_costs = []
            grads_record = []
            print '\nEpoch ', eidx, '\n'
            for idx in engine.kf_train:
                tags = [engine.train_data_ids[index] for index in idx]
                n_samples += len(tags)
                uidx += 1
                
                sess.run(tf.assign(use_noise, True))

                pd_start = time.time()
                x, mask, ctx, ctx_mask = data_engine.prepare_data(engine, tags, mode="train")
                pd_duration = time.time() - pd_start
                if x is None:
                    print 'Minibatch with zero sample under length ', maxlen
                    continue

                ud_start = time.time()
                sess.run(optimizer, feed_dict={
                                        X: x,
                                        MASK: mask,
                                        CTX: ctx,
                                        CTX_MASK: ctx_mask})
                ud_duration = time.time() - ud_start

                cost = sess.run(LOSS, feed_dict={
                                        X: x,
                                        MASK: mask,
                                        CTX: ctx,
                                        CTX_MASK: ctx_mask})
                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected in cost'
                    import pdb; pdb.set_trace()
                
                if eidx == 0:
                    train_error = cost
                else:
                    train_error = train_error * 0.95 + cost * 0.05
                train_costs.append(cost)
                # dispFreq = 1
                if np.mod(uidx, dispFreq) == 0:
                    print 'Epoch: ', eidx, \
                        ', Update: ', uidx, \
                        ', train cost mean so far: ', train_error, \
                        ', fetching data time spent (sec): ', pd_duration, \
                        ', update time spent (sec): ', ud_duration, \
                        ', save_dir: ', save_dir, '\n'
                    
                    alphas, betas = sess.run(f_alpha, feed_dict={
                                            X: x,
                                            MASK: mask,
                                            CTX: ctx,
                                            CTX_MASK: ctx_mask})
                    counts = mask.sum(0)
                    betas_mean = (betas * mask).sum(0) / counts
                    betas_mean = betas_mean.mean()
                    print 'alpha ratio %.3f, betas mean %.3f\n'%(
                        alphas.min(-1).mean() / (alphas.max(-1)).mean(), betas_mean)
                    l = 0
                    for vv in x[:, 0]:
                        print vv,
                        if vv == 0: # eos
                            break
                        if vv in engine.reverse_vocab:
                            print '(', np.round(betas[l, 0], 3), ')', engine.reverse_vocab[vv],
                        else:
                            print '(', np.round(betas[l, 0], 3), ')', 'UNK',
                        print ",",
                        l += 1
                    print '(', np.round(betas[l, 0], 3), ')\n'

                if np.mod(uidx, saveFreq) == 0:
                    pass
                # sampleFreq = 10
                if np.mod(uidx, sampleFreq) == 0:
                    sess.run(tf.assign(use_noise, False))
                    print '------------- sampling from train ----------'
                    x_s = x     # (t,m)
                    mask_s = mask   # (t,m)
                    ctx_s = ctx     # (m,28,2048)
                    ctx_mask_s = ctx_mask   # (m,28)
                    model.sample_execute(sess, engine, model_options, tfparams, f_init, f_next, x_s, ctx_s, ctx_mask_s)
                    print '------------- sampling from valid ----------'
                    idx = engine.kf_val[np.random.randint(1, len(engine.kf_val) - 1)]
                    tags = [engine.val_data_ids[index] for index in idx]
                    x_s, mask_s, ctx_s, mask_ctx_s = data_engine.prepare_data(engine, tags,"val")
                    model.sample_execute(sess, engine, model_options, tfparams, f_init, f_next, x_s, ctx_s, ctx_mask_s)
                    print ""
                    return
    return

def train_util(params):
    save_dir = params['save_dir']
    print('current save dir : '+save_dir)
    utils.create_dir_if_not_exist(save_dir)
    config_save_path = save_dir+"model_config.json"
    print('saving model config into %s' % config_save_path)
    utils.write_to_json(params, config_save_path)
    t0 = time.time()
    print('training an attention model')
    train(params, **params)
    print('training time in total %.4f sec' % (time.time() - t0))

if __name__ == '__main__':
    model_options = utils.load_default_params()
    train_util(model_options)

