import os
import keras.backend as K
from keras import initializers
from keras.engine.topology import Layer
from keras.layers import Input, Embedding, SpatialDropout1D, CuDNNGRU, CuDNNLSTM, GlobalMaxPooling1D, \
    concatenate, \
    multiply, subtract, add, TimeDistributed, BatchNormalization, Conv1D, Activation, RepeatVector
from keras.layers import Lambda
from keras.layers.wrappers import Bidirectional
from keras.losses import categorical_crossentropy
from keras.layers.core import Dense, Dropout
from keras.models import Model

from utils import load, input_dir


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
        q, k, v = x
        q = self._dropout(q)
        k = self._dropout(k)
        q_embedded = K.tanh(K.dot(q, self.Wq))
        q_embedded = K.expand_dims(q_embedded, axis=1)
        k_embedded = K.tanh(K.dot(k, self.Wk))
        joined_repr = q_embedded * k_embedded
        joined_repr = (K.dot(joined_repr, self.W))
        if self.bias:
            joined_repr += self.b
        a = K.exp(K.tanh(joined_repr))
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = v * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], self.features_dim


def get_attention_model(vqa_tr, num_hidden=512):
    embedding = load(os.path.join(input_dir, 'glove.840B.300d.npy'))
    num_rnn_unit = int(num_hidden / 2)
    shape_question = (vqa_tr.len_q,)
    input_question = Input(shape_question)
    embed1c = Embedding(embedding.shape[0], embedding.shape[1], weights=[embedding],
                        trainable=False, input_shape=shape_question)
    feature_q = embed1c(input_question)
    feature_q = SpatialDropout1D(0.2)(feature_q)
    feature_q = Bidirectional(CuDNNLSTM(num_rnn_unit))(
        feature_q)
    shape_image = vqa_tr.img_feature_shape
    assert len(shape_image) == 2
    input_image = Input(shape_image)
    feature_image = BatchNormalization(input_shape=shape_image)(input_image)
    feature_image = SpatialDropout1D(0.5, input_shape=shape_image)(feature_image)
    feature_q_r = RepeatVector(shape_image[0])(feature_q)
    feature_merge = concatenate([feature_q_r, feature_image], axis=-1)
    feature_merge_att = KVAttention(shape_image[0])([feature_merge, feature_image])
    feature_image_t = Dense(num_hidden, activation='tanh')(feature_merge_att)
    feature = multiply([feature_image_t, feature_q])
    feature_res = Dropout(0.5)(feature)
    gate_res = Dense(num_hidden, activation='sigmoid')(feature_res)
    feature = multiply([gate_res, feature_res])
    if vqa_tr.multi_label:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    out = Dense(vqa_tr.num_target, activation=activation)(feature)
    model = Model(inputs=[input_image, input_question], outputs=out)
    model.compile('adam', loss=categorical_crossentropy)
    model.summary()
    return model


def get_bottom_up_attention_model(vqa_tr, num_hidden=512):
    embedding = load(os.path.join(input_dir, 'glove.840B.300d.npy'))
    num_rnn_unit = int(num_hidden / 2)
    shape_question = (vqa_tr.len_q,)
    input_question = Input(shape_question)
    embed1c = Embedding(embedding.shape[0], embedding.shape[1], weights=[embedding],
                        trainable=False, input_shape=shape_question)
    feature_q = embed1c(input_question)
    feature_q = SpatialDropout1D(0.2)(feature_q)
    feature_q = Bidirectional(CuDNNLSTM(num_rnn_unit))(
        feature_q)
    shape_image = vqa_tr.img_feature_shape
    assert len(shape_image) == 2
    input_image = Input(shape_image)
    feature_image = Lambda(lambda x: K.l2_normalize(x, axis=-1), input_shape=shape_image)(input_image)
    feature_image_att = NewKVAttention(shape_image[0], drop_out=0.5, num_hid=num_hidden)([feature_q, feature_image,
                                                                                          feature_image])
    feature_image_t = Dense(num_hidden, activation='tanh')(feature_image_att)
    feature_q = Dense(num_hidden, activation='tanh')(Dropout(0.5)(feature_q))
    feature = multiply([feature_image_t, feature_q])
    feature_res = Dropout(0.5)(feature)
    feature = Dense(num_hidden, activation='relu')(feature_res)
    if vqa_tr.multi_label:
        activation = 'sigmoid'
    else:
        activation = 'softmax'
    out = Dense(vqa_tr.num_target, activation=activation)(feature)
    model = Model(inputs=[input_image, input_question], outputs=out)
    model.compile('adam', loss=categorical_crossentropy)
    model.summary()
    return model


def get_baseline_model(vqa_tr, num_hidden=512):
    embedding = load(os.path.join(input_dir, 'glove.840B.300d.npy'))
    num_rnn_unit = int(num_hidden / 2)
    shape_image = vqa_tr.img_feature_shape
    input_image = Input(shape_image)
    feature_image = Dropout(0.5, input_shape=shape_image)(input_image)
    feature_image_t = Dense(num_hidden, activation='tanh')(feature_image)
    feature_image = feature_image_t
    shape_question = (vqa_tr.len_q,)
    input_question = Input(shape_question)
    embed1c = Embedding(embedding.shape[0], embedding.shape[1], weights=[embedding],
                        trainable=False, input_shape=shape_question)
    feature_q = embed1c(input_question)
    feature_q = SpatialDropout1D(0.1)(feature_q)
    feature_q = Bidirectional(CuDNNLSTM(num_rnn_unit))(
        feature_q)
    feature = concatenate(
        [feature_image, feature_q, multiply([feature_image, feature_q]), subtract([feature_image, feature_q])])
    feature_res = Dropout(0.5)(feature)
    feature_res_t = (Dense(num_hidden * 4, activation='tanh'))(feature_res)
    feature_res = feature_res_t
    feature = add([feature, feature_res])
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
    embedding = load(os.path.join(input_dir, 'glove.840B.300d.npy'))
    shape_image = vqa_tr.img_feature_shape
    input_image = Input(shape_image)
    feature_image = BatchNormalization(input_shape=shape_image)(input_image)
    feature_image = SpatialDropout1D(0.5)(feature_image)
    feature_image = Conv1D(64, kernel_size=1)(feature_image)
    feature_image = Activation('relu')(feature_image)
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
    out = Dense(vqa_tr.num_target, activation='softmax')(feature)
    model = Model(inputs=[input_image, input_question], outputs=out)
    model.compile('adam', loss=categorical_crossentropy)
    model.summary()
    return model
