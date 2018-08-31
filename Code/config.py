
DATA_DIR = "../Data/MSVD/"
MSVD_CSV_DATA_PATH = "../Data/MSVD/MSVD_corpus.csv"
MSVD_PREPROC_CSV_DATA_PATH = "../Data/MSVD/processed_MSVD_corpus.csv"
MSVD_VIDEO_DATA_PATH = "../Data/MSVD/YouTubeClips/"
MSVD_OMMITTED_CAPS_PATH = "../Data/MSVD/MSVD_omitted_caps.txt"
MSVD_FINAL_CORPUS_PATH = "../Data/MSVD/MSVD_final_corpus.csv"
MSVD_VOCAB_PATH = '../Data/MSVD/MSVD_vocab.json'
MSVD_REVERSE_VOCAB_PATH = '../Data/MSVD/MSVD_reverse_vocab.json'
MSVD_VID_CAPS_TRAIN_PATH = '../Data/MSVD/MSVD_vid_caps_train.json'
MSVD_VID_CAPS_VAL_PATH = '../Data/MSVD/MSVD_vid_caps_val.json'
MSVD_VID_CAPS_TEST_PATH = '../Data/MSVD/MSVD_vid_caps_test.json'
TOTAL_VIDS = 1970
TRAIN_VIDS = 1200
TEST_VIDS = 670
VAL_VIDS = 100
MSVD_FINAL_CORPUS_TRAIN_PATH = "../Data/MSVD/MSVD_final_corpus_train.csv"
MSVD_FINAL_CORPUS_VAL_PATH = "../Data/MSVD/MSVD_final_corpus_val.csv"
MSVD_FINAL_CORPUS_TEST_PATH = "../Data/MSVD/MSVD_final_corpus_test.csv"
MSVD_VID_IDS_ALL_PATH = "../Data/MSVD/present_vid_ids.txt"
MSVD_VID_IDS_TRAIN_PATH = "../Data/MSVD/vid_ids_train.txt"
MSVD_VID_IDS_VAL_PATH = "../Data/MSVD/vid_ids_val.txt"
MSVD_VID_IDS_TEST_PATH = "../Data/MSVD/vid_ids_test.txt"
MSVD_FRAMES_DIR = "../Data/MSVD/Frames"
MSVD_FEATS_RESNET_DIR = "../Data/MSVD/Features/ResNet50/"
MSVD_FRAMES_FEATS_TRAIN_PATH = "../Data/MSVD/MSVD_resnet50_features_train.npy"
MSVD_FRAMES_FEATS_VAL_PATH = "../Data/MSVD/MSVD_resnet50_features_val.npy"
MSVD_FRAMES_FEATS_TEST_PATH = "../Data/MSVD/MSVD_resnet50_features_test.npy"
RESNET_FEAT_DIM = 2048
MAX_FRAMES = 360
FRAME_SPACING = 28
MSVD_DATA_IDS_TRAIN_PATH = "../Data/MSVD/data_ids_train.txt"
MSVD_DATA_IDS_VAL_PATH = "../Data/MSVD/data_ids_val.txt"
MSVD_DATA_IDS_TEST_PATH = "../Data/MSVD/data_ids_test.txt"

SAVE_DIR_PATH = "../Results/"

params = {
	'dataset_name' : 'msvd',
    'cnn_name' : 'resnet',
    'train_data_ids_path' : MSVD_DATA_IDS_TRAIN_PATH,
    'val_data_ids_path' : MSVD_DATA_IDS_VAL_PATH,
    'test_data_ids_path' : MSVD_DATA_IDS_TEST_PATH,
    'vocab_path' : MSVD_VOCAB_PATH,
    'reverse_vocab_path' : MSVD_REVERSE_VOCAB_PATH,
    'mb_size_train' : 64,
    'mb_size_test' : 128,
    'maxlen_caption' : 50,
    'train_caps_path' : MSVD_VID_CAPS_TRAIN_PATH,
    'val_caps_path' : MSVD_VID_CAPS_VAL_PATH,
    'test_caps_path' : MSVD_VID_CAPS_TEST_PATH,
    'feats_dir' : MSVD_FEATS_RESNET_DIR,
    'save_dir': SAVE_DIR_PATH,
    'word_dim' : 512,	# word embeddings size
    'ctx_dim' : 2048,	# video cnn feature dimension
    'lstm_dim' : 512,	# lstm unit size
    'patience' : 20,
    'max_epochs' : 500,
    'lrate' : 0.0001,
    'vocab_size' : 20000, # n_words
    'maxlen_caption' : 30,	# max length of the descprition
    'optimizer' : 'adadelta',
    'batch_size' : 64,	# for trees use 25
    'metric' : 'everything',	# set to perplexity on DVS # blue, meteor, or both
    'use_dropout' : False,
    'selector':True,
    'ctx2out':True,
    'prev2out':True,
}

# params = {
#         'reload_': False,
#         'from_dir': '',
#         'n_layers_out':1, # for predicting next word        
#         'n_layers_init':0, 
#         'encoder_dim': 300,
#         'prev2out':True, 
#         'ctx2out':True, 
#         'decay_c':1e-4,
#         'alpha_entropy_r': 0.,
#         'alpha_c':0.70602,
#         'selector':True,
#         'clip_c': 10.,
#         'valid_batch_size':200,
#         # in the unit of minibatches
#         'dispFreq':10,
#         'validFreq':2000,
#         'saveFreq':-1, # this is disabled, now use sampleFreq instead
#         'sampleFreq':100,
#         'K':28, # 26 when compare
#         'OutOf':None, # used to be 240, for motionfeature use 26
#         'verbose': True,
#         'debug': False,
#         }
