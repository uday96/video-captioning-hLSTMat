import time
import numpy as np
import utils, config

class Movie2Caption(object):
            
    def __init__(self, dataset_name,cnn_name,train_data_ids_path, val_data_ids_path, test_data_ids_path,
                vocab_path, reverse_vocab_path, mb_size_train, mb_size_test, maxlen_caption,
                train_caps_path, val_caps_path, test_caps_path, feats_dir):
        self.dataset_name = dataset_name 
        self.cnn_name = cnn_name
        self.train_data_ids_path = train_data_ids_path
        self.val_data_ids_path = val_data_ids_path
        self.test_data_ids_path = test_data_ids_path
        self.vocab_path = vocab_path
        self.reverse_vocab_path = reverse_vocab_path
        self.mb_size_train = mb_size_train
        self.mb_size_test = mb_size_test
        self.maxlen_caption = maxlen_caption
        self.train_caps_path = train_caps_path
        self.val_caps_path = val_caps_path
        self.test_caps_path = test_caps_path
        self.feats_dir = feats_dir
        self.load_data()
    
    def get_video_features(self, vid_id):
        if self.cnn_name == 'resnet':
            feat = np.load(self.feats_dir+vid_id+'.npy')
        else:
            raise NotImplementedError()
        return feat

    def get_cap_tokens(self, vid_id, cap_id, mode):
        if mode == "train":
            vid_caps = self.train_caps
        elif mode == "val":
            vid_caps = self.val_caps
        elif mode == "test":
            vid_caps = self.test_caps
        else:
            raise NotImplementedError()
        return vid_caps[vid_id][cap_id]

    def get_ctx_mask(self, ctx):
        if ctx.ndim == 3:
            rval = (ctx[:,:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 2:
            rval = (ctx[:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 5 or ctx.ndim == 4:
            assert self.video_feature == 'oxfordnet_conv3_512'
            # in case of oxfordnet features
            # (m, 26, 512, 14, 14)
            rval = (ctx.sum(-1).sum(-1).sum(-1) != 0).astype('int32').astype('float32')
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError()
        return rval
        
    def load_data(self):
        print('loading {}-{} features'.format(self.dataset_name,self.cnn_name))
        self.train_data_ids = utils.read_file_to_list(self.train_data_ids_path)
        self.val_data_ids = utils.read_file_to_list(self.val_data_ids_path)
        self.test_data_ids = utils.read_file_to_list(self.test_data_ids_path)
        self.train_caps = utils.read_from_json(self.train_caps_path)
        self.val_caps = utils.read_from_json(self.val_caps_path)
        self.test_caps = utils.read_from_json(self.test_caps_path)
        self.vocab = utils.read_from_json(self.vocab_path)
        self.reverse_vocab = utils.read_from_pickle(self.reverse_vocab_path)
        self.vocab_size = len(self.vocab)
        if self.cnn_name == 'resnet':
            self.ctx_dim = 2048
        else:
            raise NotImplementedError()
        self.kf_train = utils.generate_minibatch_idx(len(self.train_data_ids), self.mb_size_train)
        self.kf_val = utils.generate_minibatch_idx(len(self.val_data_ids), self.mb_size_test)   #TODO - verify test or val
        self.kf_test = utils.generate_minibatch_idx(len(self.test_data_ids), self.mb_size_test)
        
def prepare_data(engine, IDs, mode="train"):
    seqs = []
    feat_list = []
    for i, ID in enumerate(IDs):
        #print 'processed %d/%d caps'%(i,len(IDs))
        vidID, capID = ID.split('|')
        feat = engine.get_video_features(vidID)
        feat_list.append(feat)
        words = engine.get_cap_tokens(vidID, int(capID), mode)
        seqs.append([engine.vocab[w]
                     if w in engine.vocab and engine.vocab[w] < engine.vocab_size else 1 for w in words])   # 1 => UNK
    lengths = [len(s) for s in seqs]
    if engine.maxlen_caption != None:
        new_seqs = []
        new_feat_list = []
        new_lengths = []
        new_caps = []
        for l, s, y, c in zip(lengths, seqs, feat_list, IDs):
            # sequences that have length >= maxlen_caption will be thrown away 
            if l < engine.maxlen_caption:
                new_seqs.append(s)
                new_feat_list.append(y)
                new_lengths.append(l)
                new_caps.append(c)
        lengths = new_lengths
        feat_list = new_feat_list
        seqs = new_seqs
        if len(lengths) < 1:
            return None, None, None, None
    y = np.asarray(feat_list)   # shape (batch_size,n_frames=28,ctx_dim=2048)
    y_mask = engine.get_ctx_mask(y)
    n_samples = len(seqs)
    maxlen = np.max(lengths)+1
    x = np.zeros((maxlen, n_samples)).astype('int64')   # storing captions coloumn-wise , shape (max_seq_len,batch_size)
    x_mask = np.zeros((maxlen, n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.
    return x, x_mask, y, y_mask
    
def test_data_engine():
    # from sklearn.cross_validation import KFold
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
    engine = Movie2Caption(dataset_name,cnn_name,train_data_ids_path, val_data_ids_path, test_data_ids_path,
                vocab_path, reverse_vocab_path, mb_size_train, mb_size_test, maxlen_caption,
                train_caps_path, val_caps_path, test_caps_path, feats_dir)
    i = 0
    t = time.time()
    for idx in engine.kf_train:
        t0 = time.time()
        i += 1
        ids = [engine.train_data_ids[index] for index in idx]
        x, mask, ctx, ctx_mask = prepare_data(engine, ids, "train")
        print('seen %d minibatches, used time %.2f '%(i,time.time()-t0))
        if i == 10:
            break
    print('used time %.2f'%(time.time()-t))

if __name__ == '__main__':
    test_data_engine()
