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


class FeatureCache:
    def __init__(self, frame_aggregate_strategy, video_indexes, index_vid_map,
                 len_video,
                 feature_path, batch_size, frame_index_for_sub_instances,
                 lazy_load=False):
        self.frame_index_for_sub_instances = frame_index_for_sub_instances
        self.lazy_load = lazy_load
        self.batch_size = batch_size
        self._indices = None
        self.feature_path = feature_path
        self.len_video = len_video
        self._video_indexes = video_indexes
        self._index_vid_map = index_vid_map
        self._frame_aggregate_strategy = frame_aggregate_strategy
        self._video_features = None
        if not self.lazy_load:
            self.load_resource()

    def set_indices(self, indices):
        self._indices = indices

    def set_video_indexes(self, indexes):
        self._video_indexes = indexes

    def _pad_video(self, video):
        if self.len_video is None:
            pass
        new_shape = list(video.shape)
        new_shape[0] = self.len_video
        zeros = np.zeros(new_shape)
        len_to_copy = min(self.len_video, video.shape[0])
        zeros[:len_to_copy] = video[:len_to_copy]
        return zeros

    def load_resource(self):
        frame_aggregate_strategy = self._frame_aggregate_strategy
        feature_path = self.feature_path
        vids = self._index_vid_map[self._video_indexes]
        # TODO refactor resource loading
        if frame_aggregate_strategy == 'average':
            self._video_features = []

            with h5py.File(feature_path, 'r') as hf:
                for vid in vids:
                    # TODO efficient resource loading
                    video_feature = hf[vid][:]

                    self._video_features.append(np.mean(video_feature, axis=0, keepdims=True))
            self._video_features = np.concatenate(self._video_features, axis=0)

        elif frame_aggregate_strategy == 'no_aggregation':
            self._video_features = []

            with h5py.File(feature_path, 'r') as hf:
                for vid in vids:
                    video_feature = hf[vid][:]

                    self._video_features.append(np.expand_dims(self._pad_video(video_feature), axis=0))
            self._video_features = np.concatenate(self._video_features, axis=0)
        else:
            assert frame_aggregate_strategy == 'multi_instance'
            # the _vid index is not duplicated
            self._video_features = []
            index = 0
            with h5py.File(feature_path, 'r') as hf:
                for vid in vids:
                    video_feature = hf[vid][:]
                    # assert len(video_feature.shape) == 2
                    video_feature = video_feature[:, :12, :]
                    if self.len_video is None:
                        len_video = video_feature.shape[0]
                    else:
                        len_video = min(video_feature.shape[0], self.len_video)

                    t = self.frame_index_for_sub_instances[index:index + len_video]  # sorted(t[:len_video])
                    index += len_video
                    self._video_features.append(video_feature[t])
            self._video_features = np.array(np.concatenate(self._video_features, axis=0))

    def __getitem__(self, idx):

        inds = self._indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        if not self.lazy_load:
            batch_img_feature = self._video_features[inds]
        else:
            batch_img_feature = self.load_resource_for_video_indexes(inds)
        return batch_img_feature

    def load_resource_for_video_indexes(self, inds):
        vids = self._video_indexes[inds]

        frame_aggregate_strategy = self._frame_aggregate_strategy
        feature_path = self.feature_path
        vids = self._index_vid_map[vids]

        # TODO refactor resource loading
        if frame_aggregate_strategy == 'average':
            video_features = []

            with h5py.File(feature_path, 'r') as hf:
                for vid in vids:
                    # TODO efficient resource loading
                    video_feature = hf[vid][:]

                    video_features.append(np.mean(video_feature, axis=0, keepdims=True))
            video_features = np.concatenate(video_features, axis=0)

        elif frame_aggregate_strategy == 'no_aggregation':
            video_features = []

            with h5py.File(feature_path, 'r') as hf:
                for vid in vids:
                    video_feature = hf[vid][:]

                    video_features.append(np.expand_dims(self._pad_video(video_feature), axis=0))
            video_features = np.concatenate(video_features, axis=0)
        else:
            assert frame_aggregate_strategy == 'multi_instance'
            # the _vid index is already duplicated
            frame_to_use = self.frame_index_for_sub_instances[inds]

            # TODO reduce reading file times
            video_features = []
            with h5py.File(feature_path, 'r') as hf:
                for f, vid in zip(frame_to_use, vids):
                    video_feature = hf[vid][:]
                    video_feature = video_feature[:, :12, :]
                    video_features.append(video_feature[f])
            video_features = np.array(np.concatenate(video_features, axis=0))
        return video_features

    def __len__(self):
        return len(self._video_indexes)


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
        self._questions_text = new_data['question']
        self._video_ids_raw = new_data['video_id']
        self._index_vid_map = self._video_ids_raw.drop_duplicates().reset_index(drop=True)
        self._vid_index_map = pd.Series(self._index_vid_map.index.values, index=self._index_vid_map)  # vid ->index map
        self._video_indexes = self._vid_index_map[self._video_ids_raw].as_matrix().ravel()  # index of videos

        question_series = self.tok.texts_to_sequences(self._questions_text)
        self._questions = pad_sequences(question_series, maxlen=len_q)
        if not is_test:
            answer_series = new_data['answer']
            self._answers = self.label_encoder.transform(answer_series)
        else:
            self._answers = np.zeros([len(self._questions), self.num_target])
        self._frame_aggregate_strategy = frame_aggregate_strategy
        self._questions_raw = self._questions
        self._answers_raw = self._answers

        # self.index_vid_map = index_vid_map.sort_values()
        assert frame_aggregate_strategy in ['average', 'no_aggregation', 'multi_instance']
        self.num_instances, self.img_feature_shape, \
        self._len_sub_instances, frame_index_for_sub_instances = self._set_meta_data()
        self.video_features = FeatureCache(frame_aggregate_strategy, self._video_indexes,
                                           self._index_vid_map,
                                           self.len_video, self.feature_path,
                                           self.batch_size, frame_index_for_sub_instances)
        if self._frame_aggregate_strategy == 'multi_instance':
            self._handle_multi_instance()

            self.video_features.set_video_indexes(self._video_indexes)
        self._indices = np.arange(self._questions.shape[0], dtype=np.int32)
        self.video_features.set_indices(self._indices)
        assert self.num_instances == len(self._questions) == len(
            self._answers) == len(self._video_indexes) == len(self.video_features)
        if self._frame_aggregate_strategy == 'multi_instance':
            assert self.num_instances == len(frame_index_for_sub_instances)
            # caculate length of instances:

    def _set_meta_data(self):
        len_sub_instances = []
        frame_index_for_sub_instances = []
        video_feature_shape = None
        with h5py.File(self.feature_path, 'r') as hf:
            for vid in self._video_ids_raw:
                # TODO efficient resource loading
                video_feature_shape_raw = hf[vid][:].shape
                if self._frame_aggregate_strategy == 'average':
                    if video_feature_shape is None:
                        video_feature_shape = video_feature_shape_raw[1:]
                    else:
                        assert video_feature_shape == video_feature_shape_raw[1:]
                    len_sub_instances = np.ones([len(self._video_ids_raw)], dtype=np.int32)
                    break
                elif self._frame_aggregate_strategy == 'no_aggregation':
                    new_shape = list(video_feature_shape_raw)
                    new_shape[0] = self.len_video
                    new_shape = tuple(new_shape)
                    if video_feature_shape is None:
                        video_feature_shape = new_shape
                    else:
                        assert video_feature_shape == new_shape
                    len_sub_instances = np.ones([len(self._video_ids_raw)], dtype=np.int32)
                    break
                else:
                    assert self._frame_aggregate_strategy == 'multi_instance'
                    if video_feature_shape is None:
                        video_feature_shape = tuple(video_feature_shape_raw[1:])
                    else:
                        assert video_feature_shape == tuple(video_feature_shape_raw[1:])
                    t = np.random.permutation(video_feature_shape_raw[0])
                    if self.len_video is None:
                        len_video = video_feature_shape_raw[0]
                    else:
                        len_video = min(video_feature_shape_raw[0], self.len_video)

                    frame_index_for_sub_instances.append(sorted(t[:len_video]))
                    len_sub_instances.append(len_video)
        # TODO allow resetting frame index for each sub instances
        frame_index_for_sub_instances = np.concatenate(frame_index_for_sub_instances).ravel()
        return sum(len_sub_instances), video_feature_shape, len_sub_instances, frame_index_for_sub_instances

    def _handle_multi_instance(self):
        # duplicating q,a,vids
        vid_index = self._vid_index_map[self._video_ids_raw]
        num_sub_instances = self.num_instances
        new_questions = np.empty([num_sub_instances, self._questions.shape[1]], dtype=np.int32)
        new_answers = np.empty([num_sub_instances, self._answers.shape[1]], dtype=np.int32)
        new_vids = np.empty([num_sub_instances, ], dtype=np.int32)
        index_sub_i = 0
        for i, len_sub_instance in enumerate(self._len_sub_instances):
            new_questions[index_sub_i:index_sub_i + len_sub_instance] = np.repeat(
                np.expand_dims(self._questions_raw[i], axis=0),
                len_sub_instance, axis=0)
            new_answers[index_sub_i:index_sub_i + len_sub_instance] = np.repeat(
                np.expand_dims(self._answers_raw[i], axis=0),
                len_sub_instance, axis=0)
            new_vids[index_sub_i:index_sub_i + len_sub_instance] = np.repeat(
                vid_index[i],
                len_sub_instance)
            index_sub_i += len_sub_instance

        self._answers = new_answers
        self._questions = new_questions
        self._video_indexes = new_vids

    def __getitem__(self, idx):

        inds = self._indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_img_feature = self.video_features[idx]
        batch_sen_seq = self._questions[inds]

        if not self.is_test:
            batch_answer = self._answers[inds]
            return [batch_img_feature, batch_sen_seq], batch_answer
        else:
            return [batch_img_feature, batch_sen_seq]

    def __len__(self):
        return int(np.ceil(len(self._questions) / float(self.batch_size)))

    def on_epoch_end(self):
        # if self.frame_aggregate_strategy == 'single_instance' or self.frame_aggregate_strategy == 'multi_instance':
        #    self.load_resource(self.frame_aggregate_strategy, self.feature_path)
        # TODO  grouping questions of the same video and shuffle
        np.random.shuffle(self._indices)
        self.video_features.set_indices(self._indices)

    def eval_or_submit(self, predictions, output_path=None):
        assert self.is_test
        assert not self.shuffle_data
        num_question = self.raw_data.num_question
        if self._frame_aggregate_strategy == 'multi_instance':
            # turn predictions of sub instances into predictions of instances
            new_predictions = np.empty([len(self._video_ids_raw), predictions.shape[1]])
            index = 0
            for i, len_sub_instance in enumerate(self._len_sub_instances):
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
            vid = self._video_ids_raw[i]
            data_to_submit_i = [vid]
            for j in range(num_question):
                qj = self._questions_text[i + j]
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
        self.video_features = None
        self._answers = None
        self._questions = None


if __name__ == '__main__':
    seed = 123
    np.random.seed(seed)
    random.seed(seed + 1)
    video_feature_dir = 'faster_rcnn_10f'
    train_resource_path = 'input/%s/tr.h5' % video_feature_dir
    test_resource_path = 'input/%s/te.h5' % video_feature_dir
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
