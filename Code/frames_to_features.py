from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import utils, config

def get_resnet_model():
    # get pool5 layer output
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    height = 224
    width = 224
    return model, height, width

def img_to_feat(img_path,height,width,model):
    img = image.load_img(img_path, target_size=(height, width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = model.predict(x)
    return feat   

def extract_frames_equally_spaced(n_frames, how_many):
    # chunk frames into 'how_many' segments and use the first frame from each segment
    if n_frames < how_many:
        idx_taken = range(n_frames)
    else:   
        splits = np.array_split(range(n_frames), how_many)
        idx_taken = [s[0] for s in splits]
    return idx_taken

# TODO : save_diff_files - save seperately from next time
def frames_to_feat(vid_ids_path, num_vids, feats_outfname):
    model, height, width = get_resnet_model()
    vid_ids = utils.read_file_to_list(vid_ids_path)
    vid_clips_list = [vid[:-4] for vid in vid_ids]
    assert len(vid_ids)==num_vids
    all_vids_feats = {}
    for vid in vid_clips_list:
        print("extracting features from : "+vid)
        vid_frames_dir = config.MSVD_FRAMES_DIR+"/"+vid
        frames_list = utils.read_dir(vid_frames_dir)
        n_frames = len(frames_list)
        if n_frames > config.MAX_FRAMES:
            n_frames = config.MAX_FRAMES
        selected_frames = extract_frames_equally_spaced(n_frames,config.FRAME_SPACING)
        vid_feats = np.empty((0,config.RESNET_FEAT_DIM), float)
        for fid in selected_frames:
            img_path = vid_frames_dir+"/frame"+str(fid)+".jpg"
            # print("extracting features from : "+img_path)
            img_feat = img_to_feat(img_path,height,width,model)
            vid_feats = np.vstack((vid_feats,img_feat))
        all_vids_feats[vid] = vid_feats
        print(vid_feats.shape)
    print("saving features data at: "+feats_outfname)
    np.save(feats_outfname,all_vids_feats)        

if __name__ == '__main__':
    print("extracting features for train data..")
    frames_to_feat(config.MSVD_VID_IDS_TRAIN_PATH, config.TRAIN_VIDS, config.MSVD_FRAMES_FEATS_TRAIN_PATH)
    print("extracting features for val data..")
    frames_to_feat(config.MSVD_VID_IDS_VAL_PATH, config.VAL_VIDS, config.MSVD_FRAMES_FEATS_VAL_PATH)
    print("extracting features for test data..")
    frames_to_feat(config.MSVD_VID_IDS_TEST_PATH, config.TEST_VIDS, config.MSVD_FRAMES_FEATS_TEST_PATH)
    