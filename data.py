import h5py
import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from sklearn.utils import shuffle
import random
from utils import load, RawDataSet, raw_meta_train_path, raw_meta_test_path, save
import traceback


class VQADataSet(Sequence):
    def __init__(self, raw_ds,
                 tok_path='input/tok.pkl', label_encoder_path='input/label_encoder.pkl',
                 feature_path='output_tr.h5', multi_label=True, is_test=False,
                 batch_size=128, len_q=15, len_video=None, seed=123, num_class=1000,
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
        self.shuffle_data = shuffle_data
        self.feature_path = feature_path

        if multi_label:
            assert len(self.label_encoder.classes_) == num_class
        else:
            assert len(self.label_encoder.classes_) == num_class + 1
        self.num_target = len(self.label_encoder.classes_)
        meta_data = raw_ds
        if shuffle_data:
            meta_data.shuffle(seed=self.seed)
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

        # save to evaluate results
        self.questions_text = new_data['question']
        self.video_ids = new_data['video_id']

        question_series = self.tok.texts_to_sequences(self.questions_text)
        self.questions = pad_sequences(question_series, maxlen=len_q)
        if not is_test:
            answer_series = new_data['answer']
            self.answers = self.label_encoder.transform(answer_series)
        else:
            self.answers = np.zeros([len(self.questions), self.num_target])
        self.frame_aggregate_strategy = frame_aggregate_strategy
        self.questions_raw = self.questions
        self.answers_raw = self.answers
        assert frame_aggregate_strategy in ['average', 'no_aggregation', 'multi_instance']

        self.len_sub_instances = None
        self.img_feature_shape = None
        self.num_instances = self.set_num_of_instances()
        self.img_feature = None
        self.img_feature_index = None

        self.indices = None
        self.load_resource()
        # caculate length of instances:

    def set_num_of_instances(self):
        len_sub_instances = []

        with h5py.File(self.feature_path, 'r') as hf:
            for vid in self.video_ids:
                # TODO efficient resource loading
                video_feature_shape = hf[vid][:].shape
                if self.frame_aggregate_strategy == 'average':
                    self.img_feature_shape = video_feature_shape[1:]
                    len_sub_instances = np.ones([len(self.video_ids)], dtype=np.int32)
                    break
                elif self.frame_aggregate_strategy == 'no_aggregation':
                    new_shape = list(video_feature_shape)
                    new_shape[0] = self.len_video
                    self.img_feature_shape = tuple(new_shape)
                    len_sub_instances = np.ones([len(self.video_ids)], dtype=np.int32)
                    break
                else:
                    assert self.frame_aggregate_strategy == 'multi_instance'
                    self.img_feature_shape = tuple(video_feature_shape[1:])
                    if self.len_video is None:
                        len_video = video_feature_shape[0]
                    else:
                        len_video = min(video_feature_shape[0], self.len_video)
                    len_sub_instances.append(len_video)
        self.len_sub_instances = len_sub_instances
        return sum(self.len_sub_instances)

    def load_resource(self):
        frame_aggregate_strategy = self.frame_aggregate_strategy
        feature_path = self.feature_path
        # TODO refactor resource loading
        if frame_aggregate_strategy == 'average':
            self.img_feature = []

            with h5py.File(feature_path, 'r') as hf:
                for vid in self.video_ids:
                    # TODO efficient resource loading
                    video_feature = hf[vid][:]

                    self.img_feature.append(np.mean(video_feature, axis=0, keepdims=True))
            self.img_feature = np.concatenate(self.img_feature, axis=0)

        elif frame_aggregate_strategy == 'no_aggregation':
            self.img_feature = []

            with h5py.File(feature_path, 'r') as hf:
                for vid in self.video_ids:
                    video_feature = hf[vid][:]

                    self.img_feature.append(np.expand_dims(self.pad_video(video_feature), axis=0))
            self.img_feature = np.concatenate(self.img_feature, axis=0)
        else:
            assert frame_aggregate_strategy == 'multi_instance'
            # split each video into several smaller videos with same label
            questions = self.questions_raw
            answers = self.answers_raw
            self.img_feature = []
            with h5py.File(feature_path, 'r') as hf:
                for vid in self.video_ids:
                    video_feature = hf[vid][:]
                    # assert len(video_feature.shape) == 2
                    video_feature = video_feature[:, :12, :]
                    t = np.random.permutation(video_feature.shape[0])
                    if self.len_video is None:
                        len_video = video_feature.shape[0]
                    else:
                        len_video = min(video_feature.shape[0], self.len_video)

                    t = sorted(t[:len_video])
                    self.img_feature.append(video_feature[t])
            num_sub_instances = self.num_instances
            new_questions = np.empty([num_sub_instances, self.questions.shape[1]])
            new_answers = np.empty([num_sub_instances, self.answers.shape[1]])
            self.img_feature = np.array(np.concatenate(self.img_feature, axis=0))
            assert len(self.img_feature) == num_sub_instances
            index_sub_i = 0
            for i, len_sub_instance in enumerate(self.len_sub_instances):
                new_questions[index_sub_i:index_sub_i + len_sub_instance] = np.repeat(
                    np.expand_dims(questions[i], axis=0),
                    len_sub_instance, axis=0)
                new_answers[index_sub_i:index_sub_i + len_sub_instance] = np.repeat(
                    np.expand_dims(answers[i], axis=0),
                    len_sub_instance, axis=0)
                index_sub_i += len_sub_instance

            self.answers = new_answers
            self.questions = new_questions
        self.indices = np.arange(self.questions.shape[0], dtype=np.int32)

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

    def on_epoch_end(self):
        # if self.frame_aggregate_strategy == 'single_instance' or self.frame_aggregate_strategy == 'multi_instance':
        #    self.load_resource(self.frame_aggregate_strategy, self.feature_path)
        np.random.shuffle(self.indices)

    def eval_or_submit(self, predictions, output_path=None):
        assert self.is_test
        assert not self.shuffle_data
        num_question = self.raw_data.num_question
        if self.frame_aggregate_strategy == 'multi_instance':
            # turn predictions of sub instances into predictions of instances
            new_predictions = np.empty([len(self.video_ids), predictions.shape[1]])
            index = 0
            for i, len_sub_instance in enumerate(self.len_sub_instances):
                predictions_sub = predictions[index:index + len_sub_instance]
                new_predictions[i] = np.mean(predictions_sub, axis=0)
                index += len_sub_instance
            predictions = self.label_encoder.inverse_transform(new_predictions)
        else:
            predictions = self.label_encoder.inverse_transform(predictions)
        assert len(predictions) % num_question == 0
        # TODO handle submit in multi label case
        data_to_submit = []
        for i in range(0, len(predictions), num_question):
            vid = self.video_ids[i]
            data_to_submit_i = [vid]
            for j in range(num_question):
                qj = self.questions_text[i + j]
                pj = predictions[i + j]
                data_to_submit_i.append(qj)
                data_to_submit_i.append(pj)
            data_to_submit.append(data_to_submit_i)
        df = pd.DataFrame(data_to_submit)
        if output_path is not None:
            df.to_csv(output_path, header=None, index=False, encoding='utf-8')
            df = df.sample(frac=0.05, random_state=self.seed)  # for checking
            return df
        else:
            return RawDataSet(df, num_answer=1).eval_answers(self.raw_data)

    def clear(self):
        self.img_feature = None
        self.answers = None
        self.questions = None


if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)
    random.seed(seed + 1)
    video_feature = 'faster_rcnn_10f'
    train_resource_path = 'input/%s/tr.h5' % video_feature
    test_resource_path = 'input/%s/te.h5' % video_feature
    label_encoder_path = 'input/label_encoder_multi_1000.pkl'
    from utils import AnswerEncoder

    raw_ds_tr = RawDataSet(data_path=raw_meta_train_path)
    ds = VQADataSet(raw_ds_tr, label_encoder_path=label_encoder_path, len_video=2,
                    frame_aggregate_strategy='multi_instance', feature_path=train_resource_path, shuffle_data=True,
                    is_test=False)
    save(ds[0], 'batch_0_tr.pkl')
    save(ds[len(ds) - 1], 'batch_last_tr.pkl')
    ds.clear()
    # raw_ds_te = RawDataSet(data_path=raw_meta_test_path)
    # ds = VQADataSet(raw_ds_te, label_encoder_path=label_encoder_path, frame_aggregate_strategy='multi_instance',
    #               feature_path=test_resource_path, shuffle_data=False, is_test=True)
    # save(ds[0], 'batch_0_te.pkl')
    # save(ds[len(ds) - 1], 'batch_last_te.pkl')
