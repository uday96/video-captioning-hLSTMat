from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import utils, config

def feats_pca(cnn, vid_ids_path, num_vids, org_dim, new_dim):

	feat_save_path = config.MSVD_FEATS_DIR+cnn+"_pca"+str(new_dim)+"/"
	print "saving feats to :", feat_save_path
	utils.create_dir_if_not_exist(feat_save_path)

	vid_ids = utils.read_file_to_list(vid_ids_path)
	vid_clips_list = [vid[:-4] for vid in vid_ids]
	assert len(vid_ids)==num_vids

	vid_feats_all = np.empty((0,org_dim), dtype=np.float32)
	for vid in vid_clips_list:
		# print("loading features from : "+vid)
		vid_feats_path = config.MSVD_FEATS_DIR+cnn+"/"+vid+".npy"
		vid_feats = np.load(vid_feats_path)
		# print(vid_feats.shape)
		vid_feat_avg = np.mean(vid_feats, axis=0)
		# print(vid_feat_avg.shape)
		vid_feats_all = np.vstack((vid_feats_all,vid_feat_avg))

	print(vid_feats_all.shape)
	# vid_feats_scaled = StandardScaler().fit_transform(vid_feats_all)
	vid_feats_pca = PCA(n_components=new_dim).fit_transform(vid_feats_all)
	print(vid_feats_pca.shape)

	for ind in range(num_vids):
		vid = vid_clips_list[ind]
		vid_feat = vid_feats_pca[ind]
		# print("saving features from : "+vid)
		np.save(feat_save_path+vid+".npy",vid_feat)


def feats_kmeans(cnn, vid_ids_path, num_vids, org_dim, k):

	feat_save_path = config.MSVD_FEATS_DIR+cnn+"_kmeans"+str(k)+"/"
	print "saving feats to :", feat_save_path
	utils.create_dir_if_not_exist(feat_save_path)

	vid_ids = utils.read_file_to_list(vid_ids_path)
	vid_clips_list = [vid[:-4] for vid in vid_ids]
	assert len(vid_ids)==num_vids

	for vid in vid_clips_list:
		# print("loading features from : "+vid)
		vid_feats_path = config.MSVD_FEATS_DIR+cnn+"/"+vid+".npy"
		vid_feats = np.load(vid_feats_path)
		# print(vid_feats.shape)
		kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0).fit(vid_feats)
		vid_feat_kmeans = kmeans.cluster_centers_
		# print(vid_feat_kmeans.shape)
		np.save(feat_save_path+vid+".npy",vid_feat_kmeans)

if __name__ == '__main__':
	cnn = "ResNet152"
	original_dim = 2048
	reduced_dim = 512
	# print("pca - processing features from %s..."%cnn)
	# feats_pca(cnn, config.DATA_DIR+"present_vid_ids.txt", config.TOTAL_VIDS, original_dim, reduced_dim)
	k_centers = 3
	print("kmeans - processing features from %s..."%cnn)
	feats_kmeans(cnn, config.DATA_DIR+"present_vid_ids.txt", config.TOTAL_VIDS, original_dim, k_centers)