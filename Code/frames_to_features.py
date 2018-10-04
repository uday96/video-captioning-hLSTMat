from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from cnn.resnet152 import ResNet152
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import numpy as np
import utils, config

def get_ResNet50_model():
    # get pool5 layer output
    model = ResNet50(weights='imagenet', include_top=False, pooling="avg")
    height = 224
    width = 224
    return model, height, width, resnet50_preprocess_input

def get_ResNet152_model():
    # get pool5 layer output
    model = ResNet152(weights='imagenet', include_top=False, pooling="avg")
    height = 224
    width = 224
    return model, height, width, preprocess_input

def get_InceptionV3_model():
    model = InceptionV3(weights='imagenet', include_top=False, pooling="avg")
    height = 229
    width = 229
    return model, height, width, inception_v3_preprocess_input

def get_VGG19_model():
    model = VGG19(weights='imagenet', include_top=False, pooling="avg")
    height = 224
    width = 224
    return model, height, width, vgg19_preprocess_input

def img_to_feat(img_path, height, width, preprocess_input, model):
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

def frames_to_feat(cnn, vid_ids_path, num_vids):
    if cnn=="ResNet50":
        model, height, width, preprocess_input = get_ResNet50_model()
        FEAT_DIM = config.RESNET_FEAT_DIM
    elif cnn=="ResNet152":
        model, height, width, preprocess_input = get_ResNet152_model()
        FEAT_DIM = config.RESNET_FEAT_DIM
    elif cnn=="InceptionV3":
        model, height, width, preprocess_input = get_InceptionV3_model()
        FEAT_DIM = config.INCEPTION_FEAT_DIM
    elif cnn=="VGG19":
        model, height, width, preprocess_input = get_VGG19_model()
        FEAT_DIM = config.VGG_FEAT_DIM
    else:
        raise NotImplementedError()

    feat_save_path = config.MSVD_FEATS_DIR+cnn+"/"
    print "saving feats to :", feat_save_path
    utils.create_dir_if_not_exist(feat_save_path)

    vid_ids = utils.read_file_to_list(vid_ids_path)
    vid_clips_list = [vid[:-4] for vid in vid_ids]
    assert len(vid_ids)==num_vids

    for vid in vid_clips_list:
        print("extracting features from : "+vid)
        vid_frames_dir = config.MSVD_FRAMES_DIR+"/"+vid
        frames_list = utils.read_dir(vid_frames_dir)
        n_frames = len(frames_list)
        if n_frames > config.MAX_FRAMES:
            n_frames = config.MAX_FRAMES
        selected_frames = extract_frames_equally_spaced(n_frames,config.FRAME_SPACING)
        vid_feats = np.empty((0,FEAT_DIM), dtype=np.float32)
        for fid in selected_frames:
            img_path = vid_frames_dir+"/frame"+str(fid)+".jpg"
            # print("extracting features from : "+img_path)
            img_feat = img_to_feat(img_path,height,width,preprocess_input,model)
            vid_feats = np.vstack((vid_feats,img_feat))
        print(vid_feats.shape)
        np.save(feat_save_path+vid+".npy",vid_feats)

if __name__ == '__main__':
    cnn = "ResNet152"
    print("extracting features from %s..."%cnn)
    frames_to_feat(cnn, config.DATA_DIR+"present_vid_ids.txt", config.TOTAL_VIDS)
    