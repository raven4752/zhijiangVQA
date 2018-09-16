import h5py
import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from sklearn.utils import shuffle

from utils import load, RawDataSet


class ResetCallBack(Callback):
    def __init__(self, seq):
        super(ResetCallBack, self).__init__()
        self.seq = seq

    def on_epoch_begin(self, epoch, logs=None):
        self.seq.reset()


class VQADataSet(Sequence):
    def __init__(self, raw_ds,
                 tok_path='input/tok.pkl', label_encoder_path='input/label_encoder.pkl',
                 feature_path='output_tr.h5', multi_label=False, is_test=False,
                 batch_size=128, len_q=15, len_video=10, seed=123, num_class=1000,
                 frame_aggregate_strategy='average',
                 shuffle_data=True):
        self.len_video = len_video
        self.is_test = is_test
        self.batch_size = batch_size
        self.seed = seed
        self.len_q = len_q
        self.tok = load(tok_path)
        self.label_encoder = load(label_encoder_path)
        self.multi_label = multi_label
        if multi_label:
            assert len(self.label_encoder.classes_) == num_class
        else:
            assert len(self.label_encoder.classes_) == num_class + 1
        self.num_target = len(self.label_encoder.classes_)
        meta_data = raw_ds
        if shuffle_data:
            meta_data.shuffle()
        if multi_label:
            new_data = pd.DataFrame(meta_data.iter_vqa_line(), columns=['video_id', 'question', 'answer'])
        else:
            new_data = pd.DataFrame(meta_data.iter_vqa_pair(yield_vid=True, yield_question=True),
                                    columns=['video_id', 'question', 'answer'])
            if is_test:
                subsets = ['question', 'video_id']
            else:
                subsets = ['question', 'video_id', 'answer']
            new_data = new_data.drop_duplicates(subset=subsets).reset_index(drop=True)
        self.raw_data = meta_data
        self.indices = np.arange(new_data.shape[0])

        self.questions_raw = new_data['question']
        question_series = self.tok.texts_to_sequences(self.questions_raw)
        self.questions = pad_sequences(question_series, maxlen=len_q)
        video_ids = new_data['video_id']
        if not is_test:
            answer_series = new_data['answer']
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
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_img_feature = self.img_feature[inds]
        batch_sen_seq = self.questions[inds]
        if not self.is_test:
            batch_answer = self.answers[inds]
            return [batch_img_feature, batch_sen_seq], batch_answer
        else:
            return [batch_img_feature, batch_sen_seq]

    def __len__(self):
        return int(np.ceil(len(self.img_feature) / float(self.batch_size)))

    def reset(self):
        np.random.shuffle(self.indices)

    def eval_or_submit(self, predictions, output_path=None):
        assert self.is_test
        num_question = self.raw_data.num_question
        predictions = self.label_encoder.inverse_transform(predictions)
        assert len(predictions) % num_question == 0
        # TODO handle submit in multi label case
        data_to_submit = []
        for i in range(0, len(predictions), num_question):
            vid = self.video_ids[i]
            data_to_submit_i = [vid]
            for j in range(num_question):
                qj = self.questions_raw[i + j]
                pj = predictions[i + j]
                data_to_submit_i.append(qj)
                data_to_submit_i.append(pj)
            data_to_submit.append(data_to_submit_i)
        df = pd.DataFrame(data_to_submit)
        if output_path is not None:
            df = df.sample(frac=0.05, random_state=self.seed)  # for checking
            df.to_csv(output_path, header=None, index=False, encoding='utf-8')
            return df
        else:
            return RawDataSet(df, num_answer=1).eval_answers(self.raw_data)
