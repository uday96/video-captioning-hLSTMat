import string
import numpy as np
from nltk.tokenize import WordPunctTokenizer
import utils, config

def get_punctuations():
	return dict.fromkeys(string.punctuation)

def tokenize(sentence,punct_dict=None,translator=None):
	if not punct_dict:
		punct_dict = get_punctuations()
	if not translator:
		translator = string.maketrans("","")
	if not (type(sentence) is str and len(sentence)>0):
		print("Invalid description: "+str(sentence))
		return []
	tokens = [ token.translate(translator, string.punctuation) for token in WordPunctTokenizer().tokenize(sentence.lower())]
	filter_tokens = [token for token in tokens if (len(token)>0 and token not in punct_dict)] 
	return filter_tokens, (' '.join(filter_tokens)).strip()

def gen_vocab(df,whichdata):
	if whichdata == "test":
		outfname = config.MURALI_MSVD_VID_CAPS_TEST_PATH
		dictsize = config.MURALI_TEST_VIDS
		capspath = config.MURALI_MSVD_CAPTIONS_TEST_PATH
	elif whichdata == "val":
		outfname = config.MURALI_MSVD_VID_CAPS_VAL_PATH
		dictsize = config.MURALI_VAL_VIDS
		capspath = None
		raise NotImplementedError()
	else:
		outfname = config.MURALI_MSVD_VID_CAPS_TRAIN_PATH
		dictsize = config.MURALI_TRAIN_VIDS
		capspath = config.MURALI_MSVD_CAPTIONS_TRAIN_PATH
	vocab = set()
	punct_dict = get_punctuations()
	translator = string.maketrans("","")
	vid_caps_dict = {}
	omitted_caps = []
	for index in range(dictsize):
		vid_id = whichdata+"_"+str(index)
		descriptions = utils.read_file_to_list(capspath+str(index)+".txt")[0].split("|")
		vid_caps = []
		for desc in descriptions:
			try:
				cap = desc.strip().encode('UTF-8')
				if len(cap) > 0:
					vid_caps.append(cap)
			except Exception as e:
				# print vid_id, " : ", desc.strip()
				omitted_caps.append(vid_id+" : "+desc.strip())
		for vid_cap in vid_caps:
			tokens, _ = tokenize(vid_cap,punct_dict,translator)
			if(vid_id in vid_caps_dict):
				vid_caps_dict[vid_id].append(tokens)
			else:
				vid_caps_dict[vid_id] = [tokens]
			if whichdata=="train":
				vocab |= set(tokens)
	print("Non-ASCII captions omitted :"+str(len(omitted_caps)))
	utils.write_to_json(vid_caps_dict,outfname)
	print("Size of "+whichdata+" vid caps dict: "+str(len(vid_caps_dict)))
	assert len(vid_caps_dict)==dictsize
	if whichdata=="train":
		vocab_list = list(vocab)
		vocab_list.sort()
		vocab_dict = {vocab_list[index]:index+2 for index in range(len(vocab_list))}
		# vocab_dict['<bos>'] = 0
		vocab_dict['<eos>'] = 0
		vocab_dict['UNK'] = 1
		vocab_rev_dict = {index+2:vocab_list[index] for index in range(len(vocab_list))}
		# vocab_rev_dict[0] = '<bos>'
		vocab_rev_dict[0] = '<eos>'
		vocab_rev_dict[1] = 'UNK'
		utils.write_to_json(vocab_dict,config.MURALI_MSVD_VOCAB_PATH)
		utils.write_to_pickle(vocab_rev_dict,config.MURALI_MSVD_REVERSE_VOCAB_PATH)
		print("Size of Vocabulary: "+str(len(vocab)))
	return vocab, vid_caps_dict, omitted_caps

def prepare_data_ids(vid_caps_path, ids_save_path):
	vid_caps_dict = utils.read_from_json(vid_caps_path)
	data_ids = []
	for vid_caps in vid_caps_dict.items():
		vid_id = vid_caps[0]
		if vid_id[-4:]==".avi":
			vid_id = vid_id[:-4]
		for seq_id in range(len(vid_caps[1])):
			data_id = vid_id+"|"+str(seq_id)
			data_ids.append(data_id)
	utils.write_list_to_file(ids_save_path,data_ids)

def save_feats(whichdata):
	feat_save_path = config.MURALI_MSVD_FEATS_DIR
	print "saving feats to :", feat_save_path
	utils.create_dir_if_not_exist(feat_save_path)
	if whichdata == "train":
		encoded_feats_path = config.MURALI_MSVD_ENCODED_FEATS_TRAIN
		dictsize = config.MURALI_TRAIN_VIDS
	elif whichdata == "test":
		encoded_feats_path = config.MURALI_MSVD_ENCODED_FEATS_TEST
		dictsize = config.MURALI_TEST_VIDS
	else:
		raise NotImplementedError()
	encoded_video = np.loadtxt(encoded_feats_path,delimiter=',')
	print(encoded_video.shape)
	num, dim = encoded_video.shape
	assert num == dictsize
	for vid_id in range(num):
		vid_feats = encoded_video[vid_id].reshape(32,1024)
		# print(vid_feats.shape)
		np.save(feat_save_path+whichdata+"_"+str(vid_id)+".npy",vid_feats)

if __name__ == '__main__':
	print("generating vocab for train data...")
	vocab, _, omitted_caps_train = gen_vocab(config.MURALI_TRAIN_VIDS,"train")
	_, _, omitted_caps_test = gen_vocab(config.MURALI_TEST_VIDS,"test")
	omitted_caps = omitted_caps_train + omitted_caps_test
	utils.write_list_to_file(config.MURALI_MSVD_OMMITTED_CAPS_PATH,omitted_caps)
	print("generating train data vid+seq ids...")
	prepare_data_ids(config.MURALI_MSVD_VID_CAPS_TRAIN_PATH, config.MURALI_MSVD_DATA_IDS_TRAIN_PATH)
	# print("generating val data vid+seq ids...")
	# prepare_data_ids(config.MURALI_MSVD_VID_CAPS_VAL_PATH, config.MURALI_MSVD_DATA_IDS_VAL_PATH)
	print("generating test data vid+seq ids...")
	prepare_data_ids(config.MURALI_MSVD_VID_CAPS_TEST_PATH, config.MURALI_MSVD_DATA_IDS_TEST_PATH)
	print("seperating train vids encoded features...")
	save_feats("train")
	print("seperating test vids encoded features...")
	save_feats("test")