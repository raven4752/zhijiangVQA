#!/usr/bin/python
import logging

logging.getLogger('tensorflow').disabled = True
import json
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
import gzip
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os
import operator
import fire
import pickle
import cv2
import os
import h5py
from  sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from itertools import chain
import joblib
from tabulate import tabulate

raw_dir = 'raw'
raw_meta_train_path = raw_dir + '/train.txt'  # cleaned
raw_meta_test_path = raw_dir + '/test.txt'
raw_train_video_path = raw_dir + '/train'
raw_test_video_path = raw_dir + '/test'
raw_dev_path = raw_dir + '/dev.txt'
raw_val_path = raw_dir + '/val.txt'
column_vid = 'video_id'
mongo_url = '114.212.84.12:27017'
mongo_db = 'MY_DB'
output_dir = 'out'


def check_data():
    print('train')
    ds = RawDataSet(data_path=raw_meta_train_path)
    for v, q, answers in ds.iter_vqa_line():
        if len(q.split()) < 3:
            print(v, q)
        for a in answers:
            if len(a.strip()) == 0 or len(a.split()) > 3:
                print(v, a)
    print('test')
    ds = RawDataSet(data_path=raw_meta_test_path)
    for v, q, answers in ds.iter_vqa_line():
        if len(q.split()) < 3:
            print(v, q)
        for a in answers:
            if not isinstance(a, int):
                print(v, a)


def v2p(path='VQADatasetA_20180815/test/', pic_dir='test_pic'

        ):
    if not os.path.exists(pic_dir):
        os.mkdir(pic_dir)
    for file in os.listdir(path):
        file_id = file.split('.')[0]
        vc = cv2.VideoCapture(path + file)  # 读入视频文件
        c = 1
        out_dir = pic_dir + '/' + file_id + '/'
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        rval = False
        if vc.isOpened():  # 判断是否正常打开
            rval, frame = vc.read()
        else:
            rval = False
            print('error reading ' + file)
        # 一般MP4每秒30帧
        timeF = 10  # 视频帧计数间隔频率
        while rval:  # 循环读取视频帧
            # print("-----"+str(c))
            rows, cols, channel = frame.shape
            frame = cv2.resize(frame, (224, 224), fx=0, fy=0, interpolation=cv2.INTER_AREA)

            if c % timeF == 0:  # 每隔timeF帧进行存储操作
                print("--write-to-file---: " + str(c))
                cv2.imwrite(out_dir + str(c) + '.jpg', frame)  # 存储为图像
            c = c + 1
            rval, frame = vc.read()
        vc.release()


class RawDataSet:
    def __init__(self, data_path=raw_val_path, num_question=5, num_answer=3):
        self.num_answer = num_answer
        self.num_question = num_question
        if isinstance(data_path, str):
            self.data = pd.read_csv(data_path, header=None)
        else:
            assert isinstance(data_path, pd.DataFrame)  # loading from data frame
            self.data = data_path  #
        line1 = self.data.iloc[0]
        assert len(line1) == num_question * (num_answer + 1) + 1
        assert not self.data.isnull().values.any()

    def gen_answers(self, answer_type='oracle'):
        answers = []
        assert answer_type in ['oracle', 'dummy']
        for _, line in self.data.iterrows():
            answer = [line[0]]
            for i in range(self.num_question):
                answer.append(line[1 + i * (self.num_answer + 1)])
                if answer_type == 'oracle':
                    answer.append(line[2 + i * (self.num_answer + 1)])
                else:
                    answer.append('idontknow')
            answers.append(answer)
        df = pd.DataFrame(answers)
        df.to_csv(raw_dir + '/' + answer_type + '.txt', index=False, header=None,
                  encoding='utf-8')

    def get_num_questions_total(self):
        return len(self.data) * self.num_question

    def iter_vqa_line(self, yield_vid=True, yield_question=True, yield_answer=True):
        for _, line in self.data.iterrows():
            vid = line[0]
            for i in range(self.num_question):
                q = (line[1 + i * (self.num_answer + 1)])
                a = line[2 + i * (self.num_answer + 1):2 + (i + 1) * (self.num_answer + 1) - 1].tolist()
                things_to_iter = []
                if yield_vid:
                    things_to_iter.append(vid)
                if yield_question:
                    things_to_iter.append(q)
                if yield_answer:
                    things_to_iter.append(a)
                if len(things_to_iter) > 1:
                    yield tuple(things_to_iter)
                else:
                    yield things_to_iter[0]

    def iter_vqa_pair(self, yield_vid=False, yield_question=False, yield_answers=True):
        for answers_may_be_with_vid_q in self.iter_vqa_line(yield_vid=yield_vid, yield_question=yield_question,
                                                            yield_answer=yield_answers):
            if isinstance(answers_may_be_with_vid_q, tuple):
                answer_list = answers_may_be_with_vid_q[-1]
                others_to_yield = answers_may_be_with_vid_q[:-1]
            else:
                answer_list = answers_may_be_with_vid_q
                others_to_yield = []
            for a in answer_list:
                if len(others_to_yield) > 0:
                    yield tuple(others_to_yield + (a,))
                else:
                    yield a

    def eval_answers(self, raw_ds, debug=False):
        assert self.num_answer == 1
        assert self.get_num_questions_total() == raw_ds.get_num_questions_total()
        num_total = self.get_num_questions_total()
        num_acc = 0
        errors = []
        corrects = []
        for (vid1, q1, a), (vid2, q2, a2) in zip(self.iter_vqa_line(),
                                                 raw_ds.iter_vqa_line()):
            assert vid1 == vid2
            assert q1 == q2
            assert len(a) == 1
            if a[0] in a2:
                num_acc += 1
                corrects.append([vid1, q1, a, a2])
            else:
                errors.append([vid1, q1, a, a2])
        if not debug:
            return num_acc / num_total
        else:
            return num_acc / num_total, errors, corrects

    def split_dev_val(self, val_size=0.1, seed=123):
        data_dev, data_val = train_test_split(self.data, test_size=val_size, random_state=seed)
        return RawDataSet(data_dev, self.num_question, self.num_answer), RawDataSet(data_val, self.num_question,
                                                                                    self.num_answer)

    def split_dev_val_iter(self, val_size=0.1, seed=123, num_repeat=1):
        for i in range(num_repeat):
            yield self.split_dev_val(val_size, seed + i)

    def cv_iter(self, num_repeat=10, seed=123, yield_test_set=True):
        kfold = KFold(num_repeat, shuffle=True, random_state=seed)
        for tr_index, te_index in kfold.split(self.data):
            data_tr = self.data.loc[tr_index]
            data_te = self.data.loc[te_index]
            if yield_test_set:
                yield RawDataSet(data_tr, self.num_question, self.num_answer), RawDataSet(data_te, self.num_question,
                                                                                          self.num_answer)
            else:
                yield RawDataSet(data_tr, self.num_question, self.num_answer)

    def shuffle(self, seed=123):
        self.data = self.data.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def dev_val_split_raw(data_path=raw_meta_train_path, val_size=0.1, dev_path=raw_dev_path,
                      val_path=raw_val_path, seed=233):
    """
    split raw meta data into dev set and val set
    :return:
    """
    data = pd.read_csv(data_path, header=None)
    data_dev, data_val = train_test_split(data, test_size=val_size, random_state=seed)
    data_dev.to_csv(dev_path, index=False, header=None, encoding='utf-8')
    data_val.to_csv(val_path, index=False, header=None, encoding='utf-8')


def eval_submits(prediction_path, target_path=raw_val_path):
    try:
        return prediction_path.eval_answers(target_path)
    except:
        p = RawDataSet(prediction_path, num_answer=1)
        t = RawDataSet(target_path)
        return p.eval_answers(t)


def dump_meta_data(raw_path=raw_meta_train_path,
                   feature_path='input/faster_rcnn_10f/tr.h5'):
    # TODO efficient reading meta data
    shape = {}
    meta_data = RawDataSet(raw_path)

    new_data = pd.DataFrame(meta_data.iter_vqa_line(), columns=['video_id', 'question', 'answer'])
    video_ids_raw = new_data['video_id']
    video_ids_raw = video_ids_raw.drop_duplicates().reset_index(drop=True)
    with h5py.File(feature_path, 'r+') as hf:
        for vid in video_ids_raw:
            video_feature_shape_raw = hf[vid][:].shape
            shape[vid] = video_feature_shape_raw
    output_path = feature_path.replace('.h5', '.pkl')

    save(shape, output_path)


def pickle_h5(raw_path=raw_meta_train_path,
              feature_path='input/faster_rcnn_10f/tr.h5', channel_width=12, ):
    # TODO efficient reading meta data
    meta_data = RawDataSet(raw_path)

    new_data = pd.DataFrame(meta_data.iter_vqa_line(), columns=['video_id', 'question', 'answer'])
    video_ids_raw = new_data['video_id']
    video_ids_raw = video_ids_raw.drop_duplicates().reset_index(drop=True)
    video_features = {}
    with h5py.File(feature_path, 'r+') as hf:
        for vid in video_ids_raw:
            video_feature = hf[vid][:]
            # TODO remove ugly hack
            if len(video_feature.shape) == 3:
                video_feature = video_feature[:, :channel_width, :].astype(np.float32)
            else:
                video_feature = video_feature.astype(np.float32)
            video_features[vid] = video_feature
    output_path = feature_path.replace('.h5', '_compact.pkl')
    save(video_features, output_path, use_joblib=True)


def count_freq(sent_list):
    answer_map = {}
    num = 0
    # create answers list
    for e in sent_list:
        num += 1
        if e in answer_map:
            answer_map[e] = answer_map[e] + 1
        else:
            answer_map[e] = 1
    return answer_map, num


def report_freq(sent_list, min_freq=0, report_interval=100, print_raw=True):
    answer_map, num = count_freq(sent_list)
    print('total %d' % num)
    sorted_x = sorted(answer_map.items(), key=operator.itemgetter(1), reverse=True)
    filtered_x = []
    for (item, freq) in sorted_x:
        if freq > min_freq:
            filtered_x.append((item, freq))
    if print_raw:
        print(filtered_x[:200])
    count = 0
    for i in range(1, len(filtered_x) + 1):

        count += filtered_x[i - 1][1]
        if i % report_interval == 0:
            print(i, count / num)
    print(len(filtered_x), count / num)
    return filtered_x


def report_len(sent_list):
    lens = []
    for i, sent in enumerate(sent_list):
        lens.append(len(sent.split()))
    print(np.histogram(lens))


def stats(train_path=raw_meta_train_path, test_path=raw_meta_test_path):
    ds_tr = RawDataSet(train_path)
    ds_te = RawDataSet(test_path)
    print('len train')
    report_len(ds_tr.iter_vqa_line(yield_vid=False, yield_answer=False))
    print('len test')
    report_len(ds_te.iter_vqa_line(yield_vid=False, yield_answer=False))
    print('freq answer')
    report_freq(ds_tr.iter_vqa_pair())


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'AnswerEncoder':
            return AnswerEncoder
        return super().find_class(module, name)


def load(filename, use_joblib=False):
    print('loading from %s' % filename)
    if filename.endswith('.npy'):
        obj = np.load(filename)
    elif use_joblib:
        obj = joblib.load(filename)
    else:
        with gzip.open(filename, 'rb') as f:
            return CustomUnpickler(f).load()
            # obj = pickle.load(f)
    return obj


def save(obj, filename, save_npy=True, use_joblib=False):
    if type(obj) is np.ndarray and save_npy:
        np.save(filename, obj)
    elif use_joblib:
        joblib.dump(obj, filename, compress=3)
    else:

        with gzip.open(filename, 'wb') as f:
            pickle.dump(obj, file=f, protocol=0)


def fit_tokenizer(train_path=raw_meta_train_path, test_path=raw_meta_test_path, output_path='input/tok.pkl'):
    ds_tr = RawDataSet(train_path)
    ds_te = RawDataSet(test_path)

    texts = chain(ds_tr.iter_vqa_line(yield_vid=False, yield_answer=False),
                  ds_te.iter_vqa_line(yield_vid=False, yield_answer=False),
                  ds_tr.iter_vqa_pair())
    tok = Tokenizer()
    tok.fit_on_texts(texts)
    save(tok, output_path)


def read_embedding(embedding_file):
    """
    read embedding file into a dictionary
    each line of the embedding file should in the format like  word 0.13 0.22 ... 0.44
    :param embedding_file: path of the embedding.
    :return: a dictionary of word to its embedding (numpy array)
    """
    embeddings_index = {}
    embedding_size = None

    with open(embedding_file, mode='r', encoding='utf-8') as f:
        for line in f:

            values = line.rstrip().rsplit(' ')
            if len(values) > 2:
                try:
                    assert embedding_size is None or len(values) == embedding_size + 1
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embedding_size = coefs.shape[0]
                    embeddings_index[word] = coefs
                except:
                    print(values[0])

    return embeddings_index, embedding_size


def embedding2numpy(embedding_path='input/glove.840B.300d.txt', tok_path='input/tok.pkl',
                    ):
    output_path = embedding_path.replace('txt', 'npy')
    embedding_index, embedding_size = read_embedding(embedding_path)
    tok = load(tok_path)
    word_index = tok.word_index
    num_words = len(word_index)
    embedding_matrix = np.zeros((num_words, embedding_size))
    oov = 0
    for word, i in word_index.items():
        if i >= num_words:
            continue
        if word in embedding_index:
            embedding_vector = embedding_index[word]
            embedding_matrix[i] = embedding_vector
        else:
            oov += 1
            embedding_matrix[i] = np.zeros((1, embedding_size))
    print('oov %d/%d' % (oov, num_words))

    save(embedding_matrix, output_path, save_npy=True)


def fit_encoder(train_path=raw_meta_train_path, output_path='input/label_encoder.pkl', num_class=1000,
                multi_label=True, unknown_answer='idontknow'):
    df_tr = RawDataSet(train_path)
    answer_tr = df_tr.iter_vqa_pair()
    le = AnswerEncoder(num_class, multi_label, unknown_answer)
    le.fit(answer_tr)
    save(le, output_path)


class AnswerEncoder:
    def __init__(self, num_class=1000, multi_label=False, unknown_answer='idontknow'):
        self.unknown_answer = unknown_answer
        self.num_class = num_class
        self.multi_label = multi_label
        self.classes_ = None
        self.encoder = None

    def fit(self, answers):
        # TODO support multi-label
        answer_map, num = count_freq(answers)
        answer_freq = sorted(answer_map.items(), key=operator.itemgetter(1), reverse=True)
        kept_answers = (list(answer_freq_pair[0] for answer_freq_pair in answer_freq[:self.num_class]))
        if self.unknown_answer == 'most_freq':
            self.unknown_answer = answer_freq[self.num_class][0]
        if self.multi_label:
            self.classes_ = kept_answers
            self.encoder = MultiLabelBinarizer(classes=self.classes_).fit([self.classes_])
        else:
            self.classes_ = kept_answers + [self.unknown_answer]
            self.encoder = LabelBinarizer().fit(self.classes_)
        return self

    def transform(self, answers):
        # TODO efficient transform
        if self.multi_label:
            cleared_answers = []
            for as_list in answers:
                new_as = []
                for a in as_list:
                    if a in self.classes_:
                        new_as.append(a)
                cleared_answers.append(new_as)
        else:
            cleared_answers = []
            for answer in answers:
                if answer in self.classes_:
                    cleared_answers.append(answer)
                else:
                    cleared_answers.append(self.unknown_answer)
        return self.encoder.transform(cleared_answers)

    def inverse_transform(self, answers):
        # TODO efficient inverse_transform
        if not self.multi_label:
            return self.encoder.inverse_transform(answers)
        else:
            # get one most possible label
            t = np.zeros_like(answers)
            gold_answer_index = np.argmax(answers, axis=1)
            t[np.arange(len(answers)), gold_answer_index] = 1
            results = self.encoder.inverse_transform(t)
            return list(result[0] for result in results)


def eval_blend(*predictions_paths):
    # load data set
    ds = None
    for predictions_path in predictions_paths:
        if ds is None:
            ds = load(predictions_path.replace('.npy', '_ds.pkl'))
            # TODO check predictions made on the same val set
            break
    assert ds is not None
    predictions = []
    for predictions_path in predictions_paths:
        predictions.append(np.expand_dims(load(predictions_path), axis=0))
    predictions = np.mean(np.concatenate(predictions, axis=0), axis=0)
    score = ds.eval_or_submit(predictions)
    return score


def blend(*predictions_paths, output_path=output_dir + '/blend.txt'):
    # load data set
    ds = None
    for predictions_path in predictions_paths:
        if ds is None:
            ds = load(predictions_path.replace('.npy', '_ds.pkl'))
            # TODO check predictions made on the same val set
            break
    assert ds is not None
    predictions = []
    for predictions_path in predictions_paths:
        predictions.append(np.expand_dims(load(predictions_path), axis=0))
    predictions = np.mean(np.concatenate(predictions, axis=0), axis=0)
    return ds.eval_or_submit(predictions, output_path=output_path)


def analyse(predictions_path):
    ds = load(predictions_path.replace('.npy', '_ds.pkl'))
    predictions = load(predictions_path)

    num_question = ds.raw_data.num_question
    predictions = ds.transform_multi_instance_prediction(predictions)
    predictions = ds.label_encoder.inverse_transform(predictions)
    assert len(predictions) % num_question == 0
    df = ds.predictions_to_df(predictions, num_question)
    raw_df = ds.raw_data
    score, errors, corrects = RawDataSet(df, num_answer=1).eval_answers(raw_df, debug=True)
    with open('errors.json', 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=4, ensure_ascii=False)
    # analyse errors
    w_questions = list(e[1] for e in errors)
    c_questions = list(e[1] for e in corrects)
    w_answers = list(e[2][0] for e in errors)
    c_answers = list(e[2][0] for e in corrects)
    g_answers = []
    for e in errors:
        g_answers.extend(set(e[3]))
    print('wrong questions')
    wqs = report_freq(w_questions, min_freq=3, print_raw=False)
    print(tabulate(wqs))
    print('correct questions')
    cqs = report_freq(c_questions, min_freq=3, print_raw=False)
    print(tabulate(cqs))
    print('hard questions')
    merged = []
    c_questions = dict(cqs)
    for (wq, fwq) in wqs:
        if wq in c_questions:
            if fwq > c_questions[wq] + 10:
                merged.append([wq, fwq, c_questions[wq]])
        elif fwq > 10:
            merged.append([wq, fwq, 0])
    print(tabulate(merged, headers=['questions', 'freq in wrong', 'freq in correct'], tablefmt="pipe"))
    print('wrong answers')
    print(tabulate(report_freq(w_answers, min_freq=4, print_raw=False)))
    print('correct answers')
    print(tabulate(report_freq(c_answers, min_freq=4, print_raw=False)))
    print('hard answers')
    print(tabulate(report_freq(g_answers, min_freq=4, print_raw=False), headers=['answer', 'freq']))
    return score


if __name__ == '__main__':
    fire.Fire()
