from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import utils

MSVD_VID_IDS_ALL_PATH = "../Data/MSVD/present_vid_ids.txt"
MSVD_FRAMES_DIR = "../Data/MSVD/Frames"
MSVD_FRAMES_FEATS_PATH = "../Data/MSVD/MSVD_resnet50_features.npy"
RESNET_FEAT_DIM = 2048
TOTAL_VIDS = 1970

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

def frame_to_feat():
    model, height, width = get_resnet_model()
    vid_ids = utils.read_file_to_list(MSVD_VID_IDS_ALL_PATH)
    vid_clips_list = [vid[:-4] for vid in vid_ids]
    assert len(vid_ids)==TOTAL_VIDS
    all_vids_feats = {}
    for vid in vid_clips_list:
        print("extracting features from : "+vid)
        vid_frames_dir = MSVD_FRAMES_DIR+"/"+vid
        frames_list = utils.read_dir(vid_frames_dir)
        n_frames = len(frames_list)
        vid_feats = np.empty((0,RESNET_FEAT_DIM), float)
        for fid in range(n_frames):
            img_path = vid_frames_dir+"/frame"+str(fid)+".jpg"
            # print("extracting features from : "+img_path)
            img_feat = img_to_feat(img_path,height,width,model)
            vid_feats = np.vstack((vid_feats,img_feat))
        all_vids_feats[vid] = vid_feats
        print(vid_feats.shape)
    print("saving features data at: "+MSVD_FRAMES_FEATS_PATH)
    np.save(MSVD_FRAMES_FEATS_PATH,all_vids_feats)        

if __name__ == '__main__':
    frame_to_feat()
    