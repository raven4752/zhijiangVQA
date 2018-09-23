#!/usr/bin/python
from utils import *
import sacred
from sacred.observers import MongoObserver
from numpy.random import permutation
from sklearn import svm, datasets
from sacred import Experiment
import numpy as np
from data import VQADataSet
from model import get_bottom_up_attention_model, get_baseline_model
import datetime
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tensorflow import set_random_seed
import random

ex = Experiment('vqa')
ex.observers.append(MongoObserver.create(url=mongo_url,
                                         db_name=mongo_db))


@ex.config
def cfg():
    protocol = 'val'
    num_repeat = 1
    multi_label = True
    num_class = 500  # num of candidate answers
    len_q = 15  # length of question
    batch_size = 128
    test_batch_size = 128
    epochs = 20
    seed = 123
    lazy_load = True
    output_dir = 'out'
    frame_aggregate_strategy = 'multi_instance'
    if multi_label:
        label_encoder_path = 'input/label_encoder_multi_' + str(num_class) + '.pkl'
    else:
        label_encoder_path = 'input/label_encoder_' + str(num_class) + '.pkl'
    video_feature = 'faster_rcnn_10f'
    train_resource_path = 'input/%s/tr.h5' % video_feature
    test_resource_path = 'input/%s/te.h5' % video_feature
    artifact_dir = 'out'
    len_video = 2
    model_params = {
        'num_dense_image': 512
    }
    model_type = 'bottom-up-attention'


@ex.automain
def run(protocol, num_repeat, multi_label, num_class, len_q, batch_size, test_batch_size, epochs, seed, output_dir,
        artifact_dir,
        label_encoder_path, frame_aggregate_strategy, train_resource_path, test_resource_path, len_video, model_params,
        model_type, lazy_load):
    assert protocol in ['val', 'cv', 'submit', 'cv_submit']
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    if model_type == 'baseline':
        model_func = get_baseline_model
    else:
        assert model_type == 'bottom-up-attention'
        model_func = get_bottom_up_attention_model

    np.random.seed(seed)
    random.seed(seed + 1)
    set_random_seed(seed + 2)
    raw_ds_tr = RawDataSet(data_path=raw_meta_train_path)
    results = []
    output_path = None
    now = datetime.datetime.now()
    cur_time = now.strftime('%m_%d_%H_%M')
    out_file_name = cur_time + '_' + protocol
    if protocol in ['val', 'cv']:
        eval = True
    else:
        eval = False
    if not eval:
        raw_ds_te = RawDataSet(data_path=raw_meta_test_path)
        output_path = os.path.join(output_dir, out_file_name + '.txt')

    else:
        test_resource_path = train_resource_path

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(artifact_dir):
        os.mkdir(artifact_dir)
    if protocol == 'submit':

        def single_iter():
            yield raw_ds_tr, raw_ds_te

        valid_iter = single_iter()
    elif protocol == 'val':
        valid_iter = raw_ds_tr.split_dev_val_iter(seed=seed, num_repeat=num_repeat)
    elif protocol == 'cv_submit':
        valid_iter = raw_ds_tr.cv_iter(seed=seed, num_repeat=num_repeat, yield_test_set=False)
        valid_iter = ((tr, raw_ds_te) for tr in valid_iter)
    else:
        valid_iter = raw_ds_tr.cv_iter(seed=seed, num_repeat=num_repeat)
        # TODO eval blending predictions performance on cv
    predictions = []
    for raw_ds_tr, raw_ds_te in valid_iter:
        ds_tr = VQADataSet(raw_ds_tr, multi_label=multi_label, len_q=len_q, num_class=num_class,
                           batch_size=batch_size, feature_path=train_resource_path,
                           label_encoder_path=label_encoder_path,
                           frame_aggregate_strategy=frame_aggregate_strategy, len_video=len_video, lazy_load=lazy_load)

        model = model_func(ds_tr, **model_params)
        model.fit_generator(ds_tr, epochs=epochs, )
        ds_tr.clear()
        ds_te = VQADataSet(raw_ds_te, multi_label=multi_label, len_q=len_q, num_class=num_class,
                           batch_size=test_batch_size, is_test=True, shuffle_data=False,
                           feature_path=test_resource_path, label_encoder_path=label_encoder_path,
                           frame_aggregate_strategy=frame_aggregate_strategy, lazy_load=lazy_load)
        p_te = model.predict_generator(ds_te)
        predictions.append(p_te)
        if artifact_dir is not None:
            # TODO handle artifact saving in cv
            output_path_raw_prediction = os.path.join(artifact_dir, out_file_name + '.npy')
            output_path_ds = os.path.join(output_dir, out_file_name + '_ds.pkl')
            save(p_te, output_path_raw_prediction)
            ds_te.clear()
            save(ds_te, output_path_ds)
        score = ds_te.eval_or_submit(p_te, output_path=output_path)
        results.append(score)
        # load data set

    if len(results) == 1:
        results = results[0]
    else:
        if eval:
            results_arr = np.array(results)
            avg = results_arr.mean()
            std = results_arr.std()
            results.append(avg)
            results.append(std)
    if protocol == 'cv_submit':
        # TODO support ensemble of different aggregation strategies
        predictions_total = np.zeros_like(predictions[0])
        for p in predictions:
            predictions_total += p
        predictions_total /= len(predictions)
        results = ds_te.eval_or_submit(predictions_total, output_path=output_path)
    return results
