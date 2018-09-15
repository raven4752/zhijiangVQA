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
from  sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GroupShuffleSplit


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


def separate_q(path='VQADatasetA_20180815/', pic_dir_train='train_pic', pic_dir_test='test_pic',
               meta_train='train.txt', meta_test='test.txt'
               ):
    separate_q_single(path, pic_dir_train, meta_train)
    separate_q_single(path, pic_dir_test, meta_test)


def separate_q_single(path='VQADatasetA_20180815/', pic_dir='train_pic', meta_path='train.csv',
                      output_dir='input'):
    meta_train = pd.read_csv(path + meta_path, header=None)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # generate new meta data in format: image_dir, question, answer
    # multi_label?
    new_meta_datas = []
    for i in range(len(meta_train)):
        video_id, q1, a11, a12, a13, q2, a21, a22, a23, q3, a31, a32, a33, q4, a41, a42, a43, q5, a51, a52, a53 = \
            meta_train.loc[i]
        # image_dir_i = os.path.join(pic_dir, video_id)
        qlist = [q1, q2, q3, q4, q5]
        answers = [a11, a12, a13, a21, a22, a23, a31, a32, a33, a41, a42, a43, a51, a52, a53]
        for qid, q in enumerate(qlist):
            if len(q.split()) <= 3:
                print(video_id, q)
            for j in range(3):
                a = answers[qid * 3 + j]
                if not (a == '0' or a == 0 or len(a.split()) < 5):
                    print(video_id, a)

                new_meta_datas.append([video_id, q, a])

    df = pd.DataFrame(new_meta_datas, columns=['video_id', 'question', 'answer'])
    df = df.drop_duplicates()
    # df = shuffle(df)
    df = df.reset_index(drop=True)
    df.to_csv(os.path.join(output_dir, meta_path), index=False, encoding='utf-8')


def report_len(sent_list):
    lens = []
    for i, sent in enumerate(sent_list):
        lens.append(len(sent.split()))
    print(np.histogram(lens))


def count_freq(sent_list):
    answer_map = {}

    # create answers list
    for e in sent_list:
        if e in answer_map:
            answer_map[e] = answer_map[e] + 1
        else:
            answer_map[e] = 1
    return answer_map


def report_freq(sent_list):
    answer_map = count_freq(sent_list)
    sorted_x = sorted(answer_map.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_x[:200])
    count = 0
    for i in range(1000):
        count += sorted_x[i][1]
    print(count / len(sent_list))


def stats(train_path='input/train.txt', test_path='input/test.txt'):
    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)
    question_tr = df_tr['question']
    answer_tr = df_tr['answer']
    question_te = df_te['question']
    print('len train')
    report_len(question_tr.tolist())
    print('len test')
    report_len(question_te.tolist())
    print('freq answer')
    report_freq(answer_tr.tolist())


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


def filter_answer(num_answerable=1000, input_path='input/train.txt', output_path='input/train_c.txt'):
    df = pd.read_csv(input_path)
    answers = df['answer']
    answer_map = count_freq(answers)
    sorted_answer = sorted(answer_map.items(), key=operator.itemgetter(1), reverse=True)
    sorted_answer = sorted_answer[:num_answerable]
    answerable_list = pd.Series(list(a[0] for a in sorted_answer))
    answers_selected_index = answers.isin(answerable_list)
    df_new = df[answers_selected_index].reset_index(drop=True)
    df_new.to_csv(output_path, index=False, encoding='utf-8')


def fit_tokenizer(train_path='input/train.txt', test_path='input/test.txt', output_path='input/tok.pkl'):
    df_tr = pd.read_csv(train_path)
    df_te = pd.read_csv(test_path)
    question_tr = df_tr['question']
    answer_tr = df_tr['answer']
    question_te = df_te['question']
    texts = question_tr.tolist() + question_te.tolist() + answer_tr.tolist()
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


def fit_onehot(train_path='input/train_c.txt', output_path='input/label_encoder.pkl'):
    df_tr = pd.read_csv(train_path)
    answer_tr = df_tr['answer']
    le = LabelBinarizer()
    le.fit(answer_tr)
    assert len(le.classes_) == 1000
    save(le, output_path)


def split_dev_val(data_path='input/train_c.txt', dev_path='input/dev.txt', val_path='input/val.txt'):
    data = pd.read_csv(data_path)
    gs=GroupShuffleSplit(n_splits=1,test_size=0.1,random_state=233)
    dev_vid, val_vid = next(gs.split(data,groups=data['video_id']))
    data_dev = data.iloc[dev_vid]
    data_val = data.iloc[val_vid]
    data_dev.to_csv(dev_path, index=False, encoding='utf-8')
    data_val.to_csv(val_path,  index=False, encoding='utf-8')


def pipline():
    separate_q()
    stats()
    fit_onehot()
    fit_tokenizer()
    filter_answer()
    embedding2numpy()


if __name__ == '__main__':
    fire.Fire()
"""len train (array([   20,   562,  5723, 12841,  3931,  5405,  2212,   307,    63, 7]), array([ 1. ,  2.8,  4.6,  
6.4,  8.2, 10. , 11.8, 13.6, 15.4, 17.2, 19. ])) 
len test (array([ 100,  362,  923,  913, 1264,  271,  283,   32,   
17,    5]), array([ 3. ,  4.5,  6. ,  7.5,  9. , 10.5, 12. , 13.5, 15. , 16.5, 18. ])) 
freq answer 
[('standing', 
2128), ('indoor', 982), ('black', 871), ('sitting', 823), ('white', 780), ('blue', 639), ('in house', 441), ('red', 
411), ('residence', 383), ('drinking water', 277), ('green', 260), ('chair', 249), ('cup', 239), ('clothes', 231), 
('yellow', 230), ('table', 217), ('kitchen', 198), ('living room', 197), ('eating', 196), ('computer', 184), 
('dark blue', 177), ('grey', 174), ('on table', 168), ('book', 161), ('milk white', 153), ('in living room', 148), 
('creamy white', 147), ('gray', 145), ('talking', 143), ('bedroom', 142), ('purple', 140), ('speaking', 134), 
('brown', 131), ('pink', 121), ('walking', 120), ('inside house', 114), ('glasses', 113), ('in bedroom', 112), 
('pillow', 112), ('pot', 107), ('light blue', 99), ('in kitchen', 99), ('hat', 99), ('in room', 98), ('mirror', 96), 
('short hair', 96), ('bottle', 96), ('desk', 94), ('opening door', 92), ('sneezing', 92), ('sofa', 91), ('indoors', 
90), ('bright red', 90), ('cabinet', 90), ('aterrimus', 89), ('tidying', 87), ('lying', 86), ('phone', 85), ('bed', 
83), ('in door', 81), ('long hair', 78), ('playing cell phone', 77), ('chatting', 76), ('outdoor', 75), ('house', 
74), ('furvous', 72), ('orange', 72), ('dark black', 70), ('watching tv', 66), ('closet', 66), ('playing computer', 
66), ('cell phone', 66), ('dark grey', 65), ('resting', 65), ('on wall', 64), ('on ground', 64), ('plate', 63), 
('refrigerator', 62), ('cooking', 62), ('two', 62), ('dark red', 61), ('dark green', 61), ('closing door', 60), 
('room', 60), ('man', 58), ('taking off clothes', 58), ('playing phone', 57), ('dancing', 56), ('singing', 56), 
('light gray', 56), ('running', 55), ('clearing up', 54), ('shoes', 53), ('car', 53), ('on stairs', 53), ('towel', 
52), ('light grey', 52), ('off-white', 52), ('door', 50), ('crimson', 50), ('bag', 50), ('sleeping', 49), 
('light green', 49), ('tv', 49), ('male', 48), ('blanket', 47), ('box', 46), ('shelf', 42), ('microphone', 42), 
('laptop', 42), ('sky blue', 42), ('grass green', 42), ('painting', 41), ('looking into mirror', 41), ('taking off 
shoes', 41), ('pure black', 41), ('mouse', 41), ('dark brown', 40), ('tree', 40), ('bowl', 40), ('sweeping floor', 
40), ('pouring water', 39), ('woman', 39), ('reading book', 39), ('opening refrigerator', 38), ('lamp', 38), 
('reading', 38), ('writing', 38), ('on sofa', 38), ('paper', 37), ('light yellow', 37), ('violet', 37), ('dark-blue', 
37), ('taking things', 37), ('dog', 37), ('on bed', 37), ('bathroom', 36), ('milky white', 36), ('pool', 35), 
('curtain', 35), ('dark gray', 34), ('washing machine', 34), ('carton', 34), ('navy blue', 33), ('television', 33), 
('in car', 33), ('beige', 33), ('royal blue', 32), ('taking clothes', 32), ('toilet', 32), ('corridor', 31), 
('looking at computer', 31), ('looking in mirror', 31), ('cyan', 31), ('eating bread', 30), ('switch', 30), ('quilt', 
30), ('stairs', 30), ('eating things', 30), ('performing', 30), ('azure', 30), ('on stage', 30), ('photo frame', 30), 
('photo', 29), ('light-blue', 29), ('playing games', 29), ('rose red', 29), ('three', 29), ('water cup', 29), 
('people', 29), ('gray white', 28), ('english', 28), ('broom', 28), ('on desk', 28), ('bucket', 28), ('guitar', 27), 
('mustache', 27), ('playing mobile phone', 26), ('female', 26), ('food', 26), ('light red', 26), ('cleaning', 26), 
('reading books', 25), ('cleaning up', 25), ('beige white', 25), ('yes', 25), ('taking cup', 25), ('putting on 
clothes', 25), ('dark purple', 25), ('left hand', 25), ('carpet', 25), ('lying down', 25), ('in bathroom', 25), 
('driving', 24), ('smalt', 24), ('on chair', 24), ('window', 24), ('public place', 23), ('on floor', 23), ('hearth', 
23)] """
