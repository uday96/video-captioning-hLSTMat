import string
import numpy as np
from nltk.tokenize import WordPunctTokenizer
import utils, config

def preproc_csv(fname,outfname):
	csv_data = utils.read_csv_data(fname)
	filtered_csv = csv_data.loc[csv_data['VideoID'].notnull() & csv_data['Description'].notnull()]
	filtered_csv.to_csv(outfname, index=False, encoding='utf-8')

def filter_clips(csv_data,vid_clips_list):
	pf = []
	mf = []
	tl = 0
	pvids = []
	for index,row in csv_data.iterrows():
		fname = str(row["VideoID"])+"_"+str(row["Start"])+"_"+str(row["End"])+".avi"
		if fname in vid_clips_list:
			if fname not in pf:
				pf.append(fname)
				tl += 1
			if row["VideoID"] not in pvids:
				pvids.append(row["VideoID"])
		else:
			if fname not in mf:
				mf.append(fname)
				tl += 1
	utils.write_list_to_file(config.DATA_DIR+"present_vid_ids.txt",pf)
	utils.write_list_to_file(config.DATA_DIR+"missing_vid_ids.txt",mf)
	utils.write_list_to_file(config.DATA_DIR+"present_csv_vid_ids.txt",pvids)
	print("Present : {}".format(len(pf)))
	print("Missing : {}".format(len(mf)))
	print("Total (from CSV): {}".format(tl))
	return pf,mf,pvids

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

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def clean_caps(cap,punct_dict,translator,omitted_caps):
	tokens, clean_cap = tokenize(cap.strip(),punct_dict,translator)
	if(not is_ascii(clean_cap)):
		omitted_caps.append(cap)
		return None
	return clean_cap

def get_vid_ids(s):	# s = '4_WZN7uW0NY_140_145.avi'
	if s[-4:]=='.avi':
		s = s[:-4]
	underscore_count = 0
	for i in range(len(s)):
		i = len(s)-1-i
		if s[i]=='_':
			underscore_count = underscore_count + 1
			if underscore_count == 2:
				return s[:i]
	print("Error parsing vid id: "+s)
	return s

def clean_caps_df(csv_data,present_vid_ids,present_vid_ids_csv):
	vid_list = list(set([get_vid_ids(s) for s in present_vid_ids]))
	assert len(vid_list)==len(present_vid_ids_csv)
	df = csv_data.loc[((csv_data['VideoID'].isin(vid_list)) & (csv_data['Language'] == 'English')) & csv_data['Description'].notnull() ]
	df.to_csv(config.MSVD_FINAL_CORPUS_PATH, index=False, encoding='utf-8')
	df = utils.read_csv_data(config.MSVD_FINAL_CORPUS_PATH)
	omitted_caps = []
	punct_dict = get_punctuations()
	translator = string.maketrans("","")
	df['Description'] = df.apply(lambda row: clean_caps(row['Description'],punct_dict,translator,omitted_caps), axis=1)
	# for index,row in df.iterrows():
	# 	# row['Description'] = row['Description'].decode('ascii','ignore').strip()
	# 	# if(len(row['Description'])==0):
	# 	# 	row['Description'] = None
	# 	tokens, clean_cap = tokenize(row['Description'].strip(),punct_dict,translator)
	# 	row['Description'] = clean_cap
	# 	if(not is_ascii(row['Description'])):
	# 		omitted_caps.append(row['Description'])
	# 		row['Description'] = None
	df = df.loc[df['Description'].notnull()]
	df.to_csv(config.MSVD_FINAL_CORPUS_PATH, index=False, encoding='utf-8')
	print("Non-ASCII captions omitted :"+str(len(omitted_caps)))
	utils.write_list_to_file(config.MSVD_OMMITTED_CAPS_PATH,omitted_caps)
	return df

def gen_vocab(df,whichdata):
	if whichdata == "test":
		outfname = config.MSVD_VID_CAPS_TEST_PATH
		dictsize = config.TEST_VIDS
	elif whichdata == "val":
		outfname = config.MSVD_VID_CAPS_VAL_PATH
		dictsize = config.VAL_VIDS
	else:
		outfname = config.MSVD_VID_CAPS_TRAIN_PATH
		dictsize = config.TRAIN_VIDS
	vocab = set()
	punct_dict = get_punctuations()
	translator = string.maketrans("","")
	vid_caps_dict = {}
	for index, row in df.iterrows():
		vid_id = str(row["VideoID"])+"_"+str(row["Start"])+"_"+str(row["End"])
		tokens, _ = tokenize(row["Description"],punct_dict,translator)
		if(vid_id in vid_caps_dict):
			vid_caps_dict[vid_id].append(tokens)
		else:
			vid_caps_dict[vid_id] = [tokens]
		if whichdata=="train":
			vocab |= set(tokens)
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
		utils.write_to_json(vocab_dict,config.MSVD_VOCAB_PATH)
		utils.write_to_pickle(vocab_rev_dict,config.MSVD_REVERSE_VOCAB_PATH)
		print("Size of Vocabulary: "+str(len(vocab)))
	return vocab, vid_caps_dict

def filter_df(csv_data,vid_ids,outfname):
	vid_ids_dict = dict.fromkeys(vid_ids)
	indices = []
	for index,row in csv_data.iterrows():
		vid = str(row["VideoID"])+"_"+str(row["Start"])+"_"+str(row["End"])+".avi"
		if(vid in vid_ids_dict):
			indices.append(index)
	df = csv_data.iloc[indices]
	df.to_csv(outfname, index=False, encoding='utf-8')
	return df

def split_data(csv_data):
	vid_ids = utils.read_file_to_list(config.MSVD_VID_IDS_ALL_PATH)
	assert len(vid_ids)==config.TOTAL_VIDS
	utils.shuffle_array(vid_ids)
	train_ids = vid_ids[0:1200]
	val_ids = vid_ids[1200:1300]
	test_ids = vid_ids[1300:1970]
	assert len(train_ids)==config.TRAIN_VIDS
	assert len(val_ids)==config.VAL_VIDS
	assert len(test_ids)==config.TEST_VIDS
	utils.write_list_to_file(config.MSVD_VID_IDS_TRAIN_PATH,train_ids)
	utils.write_list_to_file(config.MSVD_VID_IDS_VAL_PATH,val_ids)
	utils.write_list_to_file(config.MSVD_VID_IDS_TEST_PATH,test_ids)
	train_df = filter_df(csv_data,train_ids,config.MSVD_FINAL_CORPUS_TRAIN_PATH)
	val_df = filter_df(csv_data,val_ids,config.MSVD_FINAL_CORPUS_VAL_PATH)
	test_df = filter_df(csv_data,test_ids,config.MSVD_FINAL_CORPUS_TEST_PATH)
	return train_df, val_df, test_df

def save_diff_files(vid_feats_path, feat_save_path):
	vid_feats = np.load(vid_feats_path)
	for feat in vid_feats[()].items():
		vid_id = feat[0]	#video ID
		vid_ctx = feat[1]	#video feature (28,2048) for resnet
		np.save(feat_save_path+vid_id+".npy",vid_ctx)

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

if __name__ == '__main__':
	has_ids_list = True
	seperate_ctx = False
	print("removing empty lines in original corpus...")
	preproc_csv(config.MSVD_CSV_DATA_PATH,config.MSVD_PREPROC_CSV_DATA_PATH)
	print("loading proccessed corpus...")
	csv_data = utils.read_csv_data(config.MSVD_PREPROC_CSV_DATA_PATH)
	print("reading video clips ids...")
	if not has_ids_list:
		vid_ids_list = utils.read_dir_files(config.MSVD_VIDEO_DATA_PATH)	# read dataset vid ids from video clips directory
	else:
		vid_ids_list = utils.read_file_to_list(config.DATA_DIR+"present_vid_ids.txt")	# read dataset vid ids from text file
	assert len(vid_ids_list)==config.TOTAL_VIDS
	print("filtering clips in df...")
	present_vid_ids, missing_vid_ids, present_vid_ids_csv = filter_clips(csv_data,vid_ids_list)
	assert len(present_vid_ids)==config.TOTAL_VIDS
	print("saving filtered df...")
	df = clean_caps_df(csv_data,present_vid_ids,present_vid_ids_csv)
	print("loading final corpus...")
	csv_data = utils.read_csv_data(config.MSVD_FINAL_CORPUS_PATH)
	print("splitting corpus into train-val-test...")
	train_df, val_df, test_df = split_data(csv_data)
	print("generating vocab for train data...")
	vocab, _ = gen_vocab(train_df,"train")
	_, _ = gen_vocab(val_df,"val")
	_, _ = gen_vocab(test_df,"test")
	if seperate_ctx:
		print("saving train data resnet features as seperate files...")
		save_diff_files(config.MSVD_FRAMES_FEATS_TRAIN_PATH, config.MSVD_FEATS_RESNET_DIR)
		print("saving val data resnet features as seperate files...")
		save_diff_files(config.MSVD_FRAMES_FEATS_VAL_PATH, config.MSVD_FEATS_RESNET_DIR)
		print("saving test data resnet features as seperate files...")
		save_diff_files(config.MSVD_FRAMES_FEATS_TEST_PATH, config.MSVD_FEATS_RESNET_DIR)
	print("generating train data vid+seq ids...")
	prepare_data_ids(config.MSVD_VID_CAPS_TRAIN_PATH, config.MSVD_DATA_IDS_TRAIN_PATH)
	print("generating val data vid+seq ids...")
	prepare_data_ids(config.MSVD_VID_CAPS_VAL_PATH, config.MSVD_DATA_IDS_VAL_PATH)
	print("generating test data vid+seq ids...")
	prepare_data_ids(config.MSVD_VID_CAPS_TEST_PATH, config.MSVD_DATA_IDS_TEST_PATH)