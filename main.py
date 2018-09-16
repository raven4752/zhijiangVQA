#!/usr/bin/python
from utils import *
import sacred
from sacred.observers import MongoObserver
from numpy.random import permutation
from sklearn import svm, datasets
from sacred import Experiment
import numpy as np
from data import VQADataSet
from model import get_baseline_model
import datetime
from model import ResetCallBack
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tensorflow import set_random_seed

ex = Experiment('vqa')
ex.observers.append(MongoObserver.create(url=mongo_url,
                                         db_name=mongo_db))


@ex.config
def cfg():
    protocol = 'val'
    num_repeat = 1
    multi_label = True
    num_class = 1000  # num of candidate answers
    len_q = 15  # length of question
    batch_size = 128
    test_batch_size = 1024
    epochs = 1
    seed = 123
    output_dir = 'out'
    frame_aggregate_strategy='multi_instance'
    if multi_label:
        label_encoder_path = 'input/label_encoder_multi_'+str(num_class)+'.pkl'
    else:
        label_encoder_path = 'input/label_encoder_'+str(num_class)+'.pkl'
    train_resource_path = 'input/vgg_10f/tr.h5'
    test_resource_path = 'input/vgg_10f/te.h5'

@ex.automain
def run(protocol, num_repeat, multi_label, num_class, len_q, batch_size, test_batch_size, epochs, seed, output_dir,
        label_encoder_path,frame_aggregate_strategy,train_resource_path,test_resource_path):
    assert protocol in ['val', 'cv', 'submit']
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    np.random.seed(seed)
    set_random_seed(seed+2)
    raw_ds_tr = RawDataSet(data_path=raw_meta_train_path)
    results = []
    output_path = None
    now = datetime.datetime.now()
    cur_time = now.strftime('%m_%d_%H')
    if protocol == 'submit':
        raw_ds_te = RawDataSet(data_path=raw_meta_test_path)

        def single_iter():
            yield raw_ds_tr, raw_ds_te

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_path = os.path.join(output_dir, cur_time + '.txt')
        valid_iter = single_iter()
    elif protocol == 'val':
        test_resource_path = train_resource_path
        valid_iter = raw_ds_tr.split_dev_val_iter(seed=seed, num_repeat=num_repeat)
    else:
        test_resource_path = train_resource_path
        valid_iter = raw_ds_tr.cv_iter(seed=seed, num_repeat=num_repeat)
    for raw_ds_tr, raw_ds_te in valid_iter:
        ds_tr = VQADataSet(raw_ds_tr, multi_label=multi_label, len_q=len_q, num_class=num_class,
                           batch_size=batch_size, feature_path=train_resource_path,label_encoder_path=label_encoder_path,
                           frame_aggregate_strategy=frame_aggregate_strategy)
        ds_te = VQADataSet(raw_ds_te, multi_label=multi_label, len_q=len_q, num_class=num_class,
                           batch_size=test_batch_size, is_test=True, shuffle_data=False,
                           feature_path=test_resource_path,label_encoder_path=label_encoder_path,
                           frame_aggregate_strategy=frame_aggregate_strategy)
        model = get_baseline_model(ds_tr)
        model.fit_generator(ds_tr, epochs=epochs, callbacks=[ResetCallBack(ds_tr)])
        p_te = model.predict_generator(ds_te)
        if output_path is not None:
            output_path_raw_prediction = os.path.join(output_dir, cur_time + '.npy')
            save(p_te, output_path_raw_prediction)
        score = ds_te.eval_or_submit(p_te, output_path=output_path)
        results.append(score)
        # load data set
    if len(results) == 1:
        results = results[0]
    return results
