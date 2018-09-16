import logging

logging.getLogger('tensorflow').disabled = True

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
from  sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from itertools import chain

raw_dir = 'VQADatasetA_20180815'
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

    def eval_answers(self, raw_ds):
        assert self.num_answer == 1
        assert self.get_num_questions_total() == raw_ds.get_num_questions_total()
        num_total = self.get_num_questions_total()
        num_acc = 0
        for (vid1, q1, a), (vid2, q2, a2) in zip(self.iter_vqa_line(),
                                                 raw_ds.iter_vqa_line()):
            assert vid1 == vid2
            assert q1 == q2
            assert len(a) == 1
            if a[0] in a2:
                num_acc += 1
        return num_acc / num_total

    def split_dev_val(self, val_size=0.1, seed=123):
        data_dev, data_val = train_test_split(self.data, test_size=val_size, random_state=seed)
        return RawDataSet(data_dev, self.num_question, self.num_answer), RawDataSet(data_val, self.num_question,
                                                                                    self.num_answer)

    def split_dev_val_iter(self, val_size=0.1, seed=123, num_repeat=1):
        for i in range(num_repeat):
            yield self.split_dev_val(val_size, seed + i)

    def cv_iter(self, num_repeat=10, seed=123):
        kfold = KFold(num_repeat, shuffle=True, random_state=seed)
        for tr_index, te_index in kfold.split(self.data):
            data_tr = self.data.loc[tr_index]
            data_te = self.data.loc[te_index]
            yield RawDataSet(data_tr, self.num_question, self.num_answer), RawDataSet(data_te, self.num_question,
                                                                                      self.num_answer)

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


def report_freq(sent_list):
    answer_map, num = count_freq(sent_list)
    sorted_x = sorted(answer_map.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_x[:200])
    count = 0
    for i in range(1, 1001):

        count += sorted_x[i - 1][1]
        if i % 100 == 0:
            print(i, count / num)


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


def load(filename):
    print('loading from %s' % filename)
    if filename.endswith('.npy'):
        obj = np.load(filename)
    else:
        with gzip.open(filename, 'rb') as f:
            obj = pickle.load(f)
    return obj


def save(obj, filename, save_npy=True):
    if type(obj) is np.ndarray and save_npy:
        np.save(filename, obj)
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


if __name__ == '__main__':
    fire.Fire()
