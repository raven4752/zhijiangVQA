from keras.utils import Sequence
import pandas as pd
from utils import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.callbacks import Callback
from sklearn.utils import shuffle
import h5py


class CommonSequence:
    def __init__(self, x, batch_size):
        self.x1 = x
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.x1.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x1 = self.x1[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x1


class ResetCallBack(Callback):
    def __init__(self, seq):
        super(ResetCallBack, self).__init__()
        self.seq = seq

    def on_epoch_begin(self, epoch, logs=None):
        self.seq.reset()


class VQASequence(Sequence):
    def __init__(self, meta_data_path='input/train_c.txt',
                 tok_path='input/tok.pkl', label_encoder_path='input/label_encoder.pkl',
                 feature_path='output_tr.h5',
                 batch_size=128, is_test=False, len_q=15, len_video=10, seed=123,
                 frame_aggregate_strategy='average',
                 shuffle_data=True):
        self.len_video = len_video
        self.is_test = is_test
        self.batch_size = batch_size
        self.seed = seed
        self.len_q = len_q
        self.tok = load(tok_path)
        self.label_encoder = load(label_encoder_path)
        self.num_target = len(self.label_encoder.classes_)
        meta_data = pd.read_csv(meta_data_path)
        if shuffle_data:
            meta_data = shuffle(meta_data, random_state=seed).reset_index()
        self.questions_raw = meta_data['question']
        question_series = self.tok.texts_to_sequences(self.questions_raw)
        self.questions = pad_sequences(question_series, maxlen=len_q)
        video_ids = meta_data['video_id']
        if not is_test:
            answer_series = meta_data['answer']
            self.answers = self.label_encoder.transform(answer_series)
            self.video_ids = None
        else:
            self.answers = None
            self.video_ids = video_ids
        assert frame_aggregate_strategy in ['average', 'no_aggregation']
        self.frame_aggregate_strategy = frame_aggregate_strategy
        if frame_aggregate_strategy == 'average':
            self.img_feature = []

            with h5py.File(feature_path, 'r') as hf:
                for vid in video_ids:
                    self.img_feature_shape = (hf[vid][:].shape[-1],)
                    self.img_feature.append(np.mean(hf[vid][:], axis=0, keepdims=True))
            self.img_feature = np.concatenate(self.img_feature, axis=0)
        else:
            self.img_feature = []

            with h5py.File(feature_path, 'r') as hf:
                for vid in video_ids:
                    self.img_feature_shape = (self.len_video, hf[vid][:].shape[-1])
                    self.img_feature.append(np.expand_dims(self.pad_video(hf[vid][:]), axis=0))
            self.img_feature = np.concatenate(self.img_feature, axis=0)

    def pad_video(self, video):
        zeros = np.zeros(self.img_feature_shape)
        len_to_copy = min(self.len_video, video.shape[0])
        zeros[:len_to_copy] = video[:len_to_copy]
        return zeros

    def __getitem__(self, idx):
        batch_img_feature = self.img_feature[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_sen_seq = self.questions[idx * self.batch_size:(idx + 1) * self.batch_size]
        if not self.is_test:
            batch_answer = self.answers[idx * self.batch_size:(idx + 1) * self.batch_size]
            return [batch_img_feature, batch_sen_seq], batch_answer
        else:
            return [batch_img_feature, batch_sen_seq]

    def __len__(self):
        return int(np.ceil(len(self.img_feature) / float(self.batch_size)))

    def reset(self):
        self.answers = shuffle(self.answers, random_state=self.seed)
        self.questions = shuffle(self.questions, random_state=self.seed)
        self.img_feature = shuffle(self.img_feature, random_state=self.seed)

    def create_submit(self, predictions):
        assert self.is_test
        predictions = self.label_encoder.inverse_transform(predictions)
        assert len(predictions) % 5 == 0
        data_to_submit = []
        for i in range(0, len(predictions), 5):
            vid = self.video_ids[i]
            q1 = self.questions_raw[i]
            q2 = self.questions_raw[i + 1]
            q3 = self.questions_raw[i + 2]
            q4 = self.questions_raw[i + 3]
            q5 = self.questions_raw[i + 4]
            p1 = predictions[i]
            p2 = predictions[i + 1]
            p3 = predictions[i + 2]
            p4 = predictions[i + 3]
            p5 = predictions[i + 4]
            data_to_submit.append([vid, q1, p1, q2, p2, q3, p3, q4, p4, q5, p5])
        return data_to_submit
