import numpy as np
import pandas as pd
import os
import fire
import h5py
from keras.preprocessing import image
from  data import VQASequence, ResetCallBack
from utils import load
from keras.layers import Input, Embedding, SpatialDropout1D, Dropout, CuDNNGRU, CuDNNLSTM, Dense, GlobalMaxPooling1D, \
    concatenate, \
    multiply, subtract, add, TimeDistributed, BatchNormalization, Conv1D, Activation, RepeatVector
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.losses import categorical_crossentropy


def feature_capture_for_video(video_path, model):
    video_paths = os.listdir(video_path)
    num_video_vid = len(video_paths)
    # load pretrained vgg net to
    image_array = np.zeros((num_video_vid, 224, 224, 3), dtype=np.float32)
    for i, c_path in enumerate(video_paths):
        c_path = os.path.join(video_path, c_path)
        img = image.load_img(c_path)
        image_array[i] = image.img_to_array(img, data_format='channels_last')
    image_array = preprocess_input(image_array)
    feature = model.predict(image_array, batch_size=128)
    return feature


from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model


def vgg_19():
    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    model.summary()
    return model


def generate_feature_single(video_capture_dir='train_pic', meta_data_dir='input/train.txt',
                            output_path='input/tr.h5'):
    meta = pd.read_csv(meta_data_dir)
    model = vgg_19()
    video_ids = meta['video_id'].drop_duplicates().reset_index(drop=True)
    count = 0
    with h5py.File(output_path, 'w') as hf:
        for vid in video_ids:
            capture_vid_dir = os.path.join(video_capture_dir, vid)
            count += 1
            if count % 100 == 0:
                print(count)
            hf.create_dataset(vid, data=feature_capture_for_video(capture_vid_dir, model))


def gen_feature():
    generate_feature_single('train_pic', 'input/train.txt', 'input/tr.h5')
    generate_feature_single('test_pic', 'input/test.txt', 'input/te.h5')


def get_baseline_model(vqa_tr):
    embedding = load('input/glove.840B.300d.npy')

    shape_image = vqa_tr.img_feature_shape
    input_image = Input(shape_image)
    # feature_image = BatchNormalization(input_shape=shape_image)(input_image)
    feature_image = Dropout(0.5, input_shape=shape_image)(input_image)
    feature_image_t = Dense(64, activation='tanh')(feature_image)
    # gate_image = Dense(64, activation='sigmoid')(feature_image)
    # feature_image = multiply([feature_image_t, gate_image])
    feature_image = feature_image_t
    shape_question = (vqa_tr.len_q,)
    input_question = Input(shape_question)
    embed1c = Embedding(embedding.shape[0], embedding.shape[1], weights=[embedding],
                        trainable=False, input_shape=shape_question)
    feature_q = embed1c(input_question)
    feature_q = SpatialDropout1D(0.1)(feature_q)
    feature_q = Bidirectional(CuDNNLSTM(32, return_sequences=True))(feature_q)
    feature_q = GlobalMaxPooling1D()(feature_q)
    feature = concatenate(
        [feature_image, feature_q, multiply([feature_image, feature_q]), subtract([feature_image, feature_q])])
    feature_res = Dropout(0.5)(feature)
    feature_res_t = (Dense(256, activation='tanh'))(feature_res)
    # gate_res = Dense(256, activation='sigmoid')(feature_res)
    # feature_res = multiply([gate_res, feature_res_t])
    feature_res = feature_res_t
    feature = add([feature, feature_res])
    # feature =feature_image
    # feature = concatenate(
    #    [feature_image, feature_q, multiply([feature_image, feature_q]), subtract([feature_image, feature_q])])
    # feature = Dropout(0.5)(feature)

    out = Dense(vqa_tr.num_target, activation='softmax')(feature)
    model = Model(inputs=[input_image, input_question], outputs=out)
    model.compile('adam', loss=categorical_crossentropy)
    model.summary()
    return model


def get_baseline_model_seq(vqa_tr):
    embedding = load('input/glove.840B.300d.npy')

    shape_image = vqa_tr.img_feature_shape
    input_image = Input(shape_image)
    feature_image = BatchNormalization(input_shape=shape_image)(input_image)
    feature_image = SpatialDropout1D(0.5)(feature_image)  # Dropout(0.5, input_shape=shape_image)(input_image)
    feature_image = Conv1D(64, kernel_size=1)(feature_image)
    feature_image = Activation('relu')(feature_image)
    # feature_image = Dense(64, activation='relu')(feature_image)
    shape_question = (vqa_tr.len_q,)
    input_question = Input(shape_question)
    embed1c = Embedding(embedding.shape[0], embedding.shape[1], weights=[embedding],
                        trainable=False, input_shape=shape_question)
    feature_q = embed1c(input_question)
    feature_q = SpatialDropout1D(0.2)(feature_q)
    feature_q = Bidirectional(CuDNNGRU(32, return_sequences=True))(feature_q)
    feature_q = GlobalMaxPooling1D()(feature_q)
    feature_q = RepeatVector(shape_image[0])(feature_q)
    feature = concatenate([feature_image, feature_q, multiply([feature_q, feature_image])])
    feature = SpatialDropout1D(0.5)(feature)
    feature = TimeDistributed(Dense(128, activation='relu'))(feature)
    feature = GlobalMaxPooling1D()(feature)
    # feature =feature_image
    # feature = concatenate(
    #    [feature_image, feature_q, multiply([feature_image, feature_q]), subtract([feature_image, feature_q])])
    # feature = Dropout(0.5)(feature)

    out = Dense(vqa_tr.num_target, activation='softmax')(feature)
    model = Model(inputs=[input_image, input_question], outputs=out)
    model.compile('adam', loss=categorical_crossentropy)
    model.summary()
    return model


if __name__ == '__main__':
    gen_feature()  # fire.Fire()
