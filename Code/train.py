import argparse
import logging
import utils, config
import time
from model import Model
import data_engine
import tensorflow as tf

def train(model_options):
    tf.set_random_seed(1234)
    model = Model()
    dataset_name = 'msvd'
    cnn_name = 'resnet'
    train_data_ids_path = config.MSVD_DATA_IDS_TRAIN_PATH
    val_data_ids_path = config.MSVD_DATA_IDS_VAL_PATH
    test_data_ids_path = config.MSVD_DATA_IDS_TEST_PATH
    vocab_path = config.MSVD_VOCAB_PATH
    reverse_vocab_path = config.MSVD_REVERSE_VOCAB_PATH
    mb_size_train = 64
    mb_size_test = 128
    maxlen_caption = 50
    train_caps_path = config.MSVD_VID_CAPS_TRAIN_PATH
    val_caps_path = config.MSVD_VID_CAPS_VAL_PATH
    test_caps_path = config.MSVD_VID_CAPS_TEST_PATH
    feats_dir = config.MSVD_FEATS_RESNET_DIR
    print 'Loading data'
    engine = data_engine.Movie2Caption(dataset_name,cnn_name,train_data_ids_path, val_data_ids_path, test_data_ids_path,
                vocab_path, reverse_vocab_path, mb_size_train, mb_size_test, maxlen_caption,
                train_caps_path, val_caps_path, test_caps_path, feats_dir)
    model_options['dim_ctx'] = engine.ctx_dim
    model_options['vocab_size'] = engine.vocab_size
    print 'n_words:', model_options['vocab_size']
    # set test values, for debugging
    idx = engine.kf_train[0]
    x_tv, mask_tv, ctx_tv, ctx_mask_tv = data_engine.prepare_data(engine, [engine.train_data_ids[index] for index in idx], mode="train")
    print 'init params'
    t0 = time.time()
    params = model.init_params(model_options)
    # description string: #words x #samples
    x = tf.placeholder(tf.int64, shape=(None,model_options['batch_size']), name='word_seq_x')  # word seq input
    mask = tf.placeholder(tf.float32, shape=(None,model_options['batch_size']), name='word_seq_mask')
    # context: #samples x #annotations x dim
    ctx = tf.placeholder(tf.float32, shape=(model_options['batch_size'],28,model_options['ctx_dim']), name='ctx')
    ctx_mask = tf.placeholder(tf.float32, shape=(model_options['batch_size'],28), name='ctx_mask')
    # create tensorflow variables
    tfparams = utils.init_tfparams(params)
    init_state, init_memory, bo_lstm = model.build_model(tfparams, model_options, x, mask, ctx, ctx_mask)
    # Initialize all variables
    var_init = tf.global_variables_initializer()
    # Ops to save and restore all the variables.
    saver = tf.train.Saver()
    # Launch the graph
    with tf.Session() as sess:
        sess.run(var_init)
        ans = sess.run([init_state,init_memory,bo_lstm], feed_dict={
                    x: x_tv,
                    mask: mask_tv,
                    ctx: ctx_tv,
                    ctx_mask: ctx_mask_tv})
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
    train(params)
    print('training time in total %.4f sec' % (time.time() - t0))

if __name__ == '__main__':
    model_options = utils.load_default_params()
    train_util(model_options)

