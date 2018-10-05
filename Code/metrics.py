import argparse, os, pdb, sys, time
import numpy as np
import cPickle as pkl
import copy
import glob
import subprocess
from multiprocessing import Process, Queue, Manager
from collections import OrderedDict

import train
import data_engine
from cocoeval import COCOScorer
import utils, config
    
MAXLEN = 50

# Only for testing
def build_sample_pairs_test(IDs, engine, mode):
    D = OrderedDict()
    for ID in IDs:
        vidID, capID = ID.split('|')
        words = engine.get_cap_tokens(vidID, int(capID), mode)
        caption = ' '.join(words)
        D[vidID] = [{'image_id': vidID, 'caption': caption}]
    return D

def build_sample_pairs(samples, vidIDs):
    D = OrderedDict()
    for sample, vidID in zip(samples, vidIDs):
        D[vidID] = [{'image_id': vidID, 'caption': sample}]
    return D

def score_with_cocoeval(samples_valid, samples_test, engine):
    scorer = COCOScorer()
    if samples_valid:
        gts_valid = OrderedDict()
        for ID in engine.val_data_ids:
            vidID, capID = ID.split('|')
            words = engine.get_cap_tokens(vidID, int(capID), mode='val')
            caption = ' '.join(words)
            if gts_valid.has_key(vidID):
                gts_valid[vidID].append({'image_id': vidID, 'caption': caption, 'cap_id': capID})
            else:
                gts_valid[vidID] = [{'image_id': vidID, 'caption': caption, 'cap_id': capID}]
        valid_score = scorer.score(gts_valid, samples_valid, gts_valid.keys())
    else:
        valid_score = None
    if samples_test:
        gts_test = OrderedDict()
        for ID in engine.test_data_ids:
            vidID, capID = ID.split('|')
            words = engine.get_cap_tokens(vidID, int(capID), mode='test')
            caption = ' '.join(words)
            if gts_test.has_key(vidID):
                gts_test[vidID].append({'image_id': vidID, 'caption': caption, 'cap_id': capID})
            else:
                gts_test[vidID] = [{'image_id': vidID, 'caption': caption, 'cap_id': capID}]
        test_score = scorer.score(gts_test, samples_test, gts_test.keys())
    else:
        test_score = None
    return valid_score, test_score

def generate_sample_gpu_single_process(sess,
        model_type, model_archive, options, engine, model,
        f_init, f_next,
        save_dir='./samples/', beam=5,
        whichset='both'):
    
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(engine.reverse_vocab[1]
                          if w > len(engine.reverse_vocab) else engine.reverse_vocab[w])
            capsw.append(' '.join(ww))
        return capsw
    
    def sample(whichset):
        samples = []
        ctxs, ctx_masks = engine.prepare_data_for_blue(whichset)
        for i, ctx, ctx_mask in zip(range(len(ctxs)), ctxs, ctx_masks):
            print 'sampling %d/%d'%(i,len(ctxs))
            stochastic = not options['beam_search']
            if stochastic:
                kbeam = 1
            else:
                kbeam = 5
            sample, score, _, _ = model.gen_sample(sess,
                None, f_init, f_next, ctx, ctx_mask, options,
                k=kbeam, maxlen=MAXLEN, stochastic=stochastic)
            if not stochastic:
                sidx = np.argmin(score)
                sample = sample[sidx]
            else:
                sample = [sample]
            samples.append(sample)
        samples = _seqs2words(samples)
        return samples

    samples_valid = None
    samples_test = None
    if whichset == 'val' or whichset == 'both':
        print 'Valid Set...',
        samples_valid = sample('val')
        with open(save_dir+'valid_samples.txt', 'w') as f:
            print >>f, '\n'.join(samples_valid)
    if whichset == 'test' or whichset == 'both':
        print 'Test Set...',
        samples_test = sample('test')
        with open(save_dir+'test_samples.txt', 'w') as f:
            print >>f, '\n'.join(samples_test)

    if samples_valid:
        samples_valid = build_sample_pairs(samples_valid, engine.val_ids)
    if samples_test:
        samples_test = build_sample_pairs(samples_test, engine.test_ids)
    return samples_valid, samples_test

def compute_score(sess,
        model_type, model_archive, options, engine, save_dir,
        beam, n_process,
        whichset='both', on_cpu=True,
        processes=None, queue=None, rqueue=None, shared_params=None,
        one_time=False, metric=None,
        f_init=None, f_next=None, model=None):

    assert metric != 'perplexity'
    if on_cpu:
        raise NotImplementedError()
    else:
        assert model is not None
        samples_valid, samples_test = generate_sample_gpu_single_process(sess,
            model_type, model_archive,options,
            engine, model, f_init, f_next,
            save_dir=save_dir,
            beam=beam,
            whichset=whichset)
        
    valid_score, test_score = score_with_cocoeval(samples_valid, samples_test, engine)
    scores_final = {}
    scores_final['valid'] = valid_score
    scores_final['test'] = test_score
    
    if one_time:
        return scores_final
    
    return scores_final, processes, queue, rqueue, shared_params

def test_cocoeval():
    dataset_name = 'MSVD'
    cnn_name = 'ResNet152'
    train_data_ids_path = config.MSVD_DATA_IDS_TRAIN_PATH
    val_data_ids_path = config.MSVD_DATA_IDS_VAL_PATH
    test_data_ids_path = config.MSVD_DATA_IDS_TEST_PATH
    vocab_path = config.MSVD_VOCAB_PATH
    reverse_vocab_path = config.MSVD_REVERSE_VOCAB_PATH
    mb_size_train = 64
    mb_size_test = 128
    maxlen_caption = 30
    train_caps_path = config.MSVD_VID_CAPS_TRAIN_PATH
    val_caps_path = config.MSVD_VID_CAPS_VAL_PATH
    test_caps_path = config.MSVD_VID_CAPS_TEST_PATH
    feats_dir = config.MSVD_FEATS_DIR+cnn_name+"/"
    engine = data_engine.Movie2Caption(dataset_name,cnn_name,train_data_ids_path, val_data_ids_path, test_data_ids_path,
                vocab_path, reverse_vocab_path, mb_size_train, mb_size_test, maxlen_caption,
                train_caps_path, val_caps_path, test_caps_path, feats_dir)
    samples_valid = engine.val_data_ids
    samples_test = engine.test_data_ids
    samples_valid = build_sample_pairs_test(samples_valid, engine, mode="val")
    samples_test = build_sample_pairs_test(samples_test, engine, mode="test")
    valid_score, test_score = score_with_cocoeval(samples_valid, samples_test, engine)
    print valid_score

if __name__ == '__main__':
    test_cocoeval()
