import numpy as np
import pandas as pd
import os
import fire
import h5py
from keras.preprocessing import image
from  data import VQADataSet
from utils import load
from keras.layers import Input, Embedding, SpatialDropout1D, Dropout, CuDNNGRU, CuDNNLSTM, Dense, GlobalMaxPooling1D, \
    concatenate, \
    multiply, subtract, add, TimeDistributed, BatchNormalization, Conv1D, Activation, RepeatVector
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.losses import categorical_crossentropy
import keras.backend as K
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer, InputSpec
from keras.initializers import Ones, Zeros
from keras.layers import Lambda


class KVAttention(Layer):
    def __init__(self, step_dim,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(KVAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_shape_sub = input_shape[0]
        assert len(input_shape_sub) == 3

        self.W = self.add_weight((input_shape_sub[-1], 1),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 )
        self.features_dim = input_shape[1][-1]

        if self.bias:
            self.b = self.add_weight((input_shape_sub[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     )
        else:
            self.b = None

        self.built = True

    def get_config(self):
        config = {'step_dim': self.step_dim}
        base_config = super(KVAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        k, v = x
        eij = K.dot(k, self.W)
        eij = K.squeeze(eij, axis=-1)
        features_dim = self.features_dim
        if self.bias:
            eij += self.b
        eij = K.tanh(eij)
        a = K.exp(eij)
        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = v * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0][0], self.features_dim


class NewKVAttention(Layer):
    def __init__(self, step_dim, num_hid=512,
                 bias=True, drop_out=0.0, **kwargs):
        self.num_hid = num_hid
        self.drop_out = drop_out
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(NewKVAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3  # q,k,v
        input_shape_q = input_shape[0]
        input_shape_k = input_shape[1]
        input_shape_v = input_shape[2]
        assert len(input_shape_q) == 2
        assert len(input_shape_k) == 3 == len(input_shape_v)

        self.Wq = self.add_weight((input_shape_q[-1], self.num_hid),
                                  initializer=self.init,
                                  name='{}_Wq'.format(self.name),
                                  )
        self.Wk = self.add_weight([input_shape_k[-1], self.num_hid],
                                  initializer=self.init,
                                  name='{}_Wk'.format(self.name),
                                  )
        self.W = self.add_weight((self.num_hid, 1),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 )
        self.features_dim = input_shape_v[-1]

        if self.bias:
            self.b = self.add_weight((input_shape_k[1], 1),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     )
        else:
            self.b = None

        self.built = True

    def get_config(self):
        config = {'step_dim': self.step_dim, 'bias': self.bias, 'drop_out': self.drop_out, 'num_hid': self.num_hid}
        base_config = super(NewKVAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def _dropout(self, x):
        return K.in_train_phase(K.dropout(x, level=self.drop_out), x)

    def call(self, x, mask=None, training=None):
        #  W^T q * W^T k
        q, k, v = x
        q = self._dropout(q)
        k = self._dropout(k)
        q_embedded = K.tanh(K.dot(q, self.Wq))
        q_embedded = K.expand_dims(q_embedded, axis=1)
        k_embedded = K.tanh(K.dot(k, self.Wk))
        joined_repr = q_embedded * k_embedded
        # joined_repr = self._dropout(joined_repr)
        # joined_repr = K.dropout(joined_repr, level=self.drop_out)
        joined_repr = (K.dot(joined_repr, self.W))
        if self.bias:
            joined_repr += self.b
        a = K.exp(K.tanh(joined_repr))
        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = v * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0][0], self.features_dim


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


def get_bottom_up_attention_model(vqa_tr, num_hidden=512):
    embedding = load('input/glove.840B.300d.npy')
    num_rnn_unit = int(num_hidden / 2)
    shape_question = (vqa_tr.len_q,)
    input_question = Input(shape_question)
    embed1c = Embedding(embedding.shape[0], embedding.shape[1], weights=[embedding],
                        trainable=False, input_shape=shape_question)
    feature_q = embed1c(input_question)
    feature_q = SpatialDropout1D(0.2)(feature_q)
    feature_q = Bidirectional(CuDNNLSTM(num_rnn_unit))(
        feature_q)

    # feature_q = Dropout(0.5)(feature_q)
    shape_image = vqa_tr.img_feature_shape
    assert len(shape_image) == 2
    input_image = Input(shape_image)
    feature_image = Lambda(lambda x: K.l2_normalize(x, axis=-1), input_shape=shape_image)(input_image)
    # feature_image = BatchNormalization(input_shape=shape_image)(input_image)
    # feature_image = input_image
    # feature_image = Dropout(0.5, input_shape=shape_image)(feature_image)

    # feature_q_r = RepeatVector(shape_image[0])(feature_q)
    # feature_q_r = SpatialDropout1D(0.5)(feature_q_r)
    # feature_merge = concatenate([feature_q_r, feature_image], axis=-1)
    # feature_merge_att = KVAttention(shape_image[0])([feature_merge, feature_image])
    # feature_merge_att = Dropout(0.5)(feature_merge_att)
    feature_image_att = NewKVAttention(shape_image[0], drop_out=0.5, num_hid=num_hidden)([feature_q, feature_image,
                                                                                          feature_image])
    feature_image_t = Dense(num_hidden, activation='tanh')(feature_image_att)
    # gate_image = Dense(num_dense_image, activation='sigmoid')(feature_merge_att)
    # feature_image = multiply([feature_merge_att, gate_image])
    feature_q = Dense(num_hidden, activation='tanh')(Dropout(0.5)(feature_q))
    # feature_q = Dropout(0.5)(feature_q)
    # Bidirectional(CuDNNLSTM(32, return_sequences=True))(feature_q)
    # feature_q = GlobalMaxPooling1D()(feature_q)
    feature = multiply([feature_image_t, feature_q])
    # feature= concatenate(
    #    [feature_image, feature_q, multiply([feature_image, feature_q]), subtract([feature_image, feature_q])])
    feature_res = Dropout(0.5)(feature)
    # feature_res_t = (Dense(num_dense_image * 4, activation='tanh'))(feature_res)
    # gate_res = Dense(num_dense_image, activation='sigmoid')(feature_res)
    # feature_res = multiply([gate_res, feature_res])
    # feature_res = feature_res_t
    feature = feature_res
    # feature = add([feature, feature_res])
    # feature =feature_image
    # feature = concatenate(
    #    [feature_image, feature_q, multiply([feature_image, feature_q]), subtract([feature_image, feature_q])])
    # feature = Dropout(0.5)(feature)
    if vqa_tr.multi_label:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    out = Dense(vqa_tr.num_target, activation=activation)(feature)
    # out_gate = Dense(vqa_tr.num_target, activation='sigmoid')(Dropout(0.5)(out))
    # out = multiply([out, out_gate])
    model = Model(inputs=[input_image, input_question], outputs=out)
    model.compile('adam', loss=categorical_crossentropy)
    model.summary()
    return model


def get_baseline_model(vqa_tr, num_dense_image=128):
    embedding = load('input/glove.840B.300d.npy')
    num_rnn_unit = int(num_dense_image / 2)
    shape_image = vqa_tr.img_feature_shape
    input_image = Input(shape_image)
    # feature_image = BatchNormalization(input_shape=shape_image)(input_image)
    feature_image = Dropout(0.5, input_shape=shape_image)(input_image)
    feature_image_t = Dense(num_dense_image, activation='tanh')(feature_image)
    # gate_image = Dense(num_dense_image, activation='sigmoid')(feature_image)
    # feature_image = multiply([feature_image_t, gate_image])
    feature_image = feature_image_t
    shape_question = (vqa_tr.len_q,)
    input_question = Input(shape_question)
    embed1c = Embedding(embedding.shape[0], embedding.shape[1], weights=[embedding],
                        trainable=False, input_shape=shape_question)
    feature_q = embed1c(input_question)
    feature_q = SpatialDropout1D(0.1)(feature_q)
    feature_q = Bidirectional(CuDNNLSTM(num_rnn_unit))(
        feature_q)  # Bidirectional(CuDNNLSTM(32, return_sequences=True))(feature_q)
    # feature_q = GlobalMaxPooling1D()(feature_q)
    feature = concatenate(
        [feature_image, feature_q, multiply([feature_image, feature_q]), subtract([feature_image, feature_q])])
    feature_res = Dropout(0.5)(feature)
    feature_res_t = (Dense(num_dense_image * 4, activation='tanh'))(feature_res)
    # gate_res = Dense(256, activation='sigmoid')(feature_res)
    # feature_res = multiply([gate_res, feature_res_t])
    feature_res = feature_res_t
    feature = add([feature, feature_res])
    # feature =feature_image
    # feature = concatenate(
    #    [feature_image, feature_q, multiply([feature_image, feature_q]), subtract([feature_image, feature_q])])
    # feature = Dropout(0.5)(feature)
    if vqa_tr.multi_label:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    out = Dense(vqa_tr.num_target, activation=activation)(feature)

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
