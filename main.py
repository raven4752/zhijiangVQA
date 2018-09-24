#!/usr/bin/python
from utils import *
import sacred
from sacred.observers import MongoObserver
from numpy.random import permutation
from sklearn import svm, datasets
from sacred import Experiment
import numpy as np
from data import VQADataSet
from model import get_bottom_up_attention_model, get_baseline_model, get_attention_model
import datetime
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tensorflow import set_random_seed
import random
from easydict import EasyDict as edict

ex = Experiment('vqa')
ex.observers.append(MongoObserver.create(url=mongo_url,
                                         db_name=mongo_db))
ex.add_config('config.yaml')


@ex.automain
def run(protocol, num_repeat, data_opts, epochs, seed,
        output_dir, artifact_dir, model_params, model_type):
    assert protocol in ['val', 'cv', 'submit', 'cv_submit', 'cv_val']
    # setting gpu memory limit
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    if model_type == 'baseline':
        model_func = get_baseline_model
    elif model_type == 'attention':
        model_func = get_attention_model
    else:
        assert model_type == 'bottom-up-attention'
        model_func = get_bottom_up_attention_model
    # extract data options
    data_opts = edict(data_opts)
    multi_label = data_opts.multi_label
    video_feature = data_opts.video_feature
    num_class = data_opts.num_class
    train_resource_path = 'input/%s/tr.h5' % video_feature
    test_resource_path = 'input/%s/te.h5' % video_feature

    if multi_label:
        label_encoder_path = 'input/label_encoder_multi_' + str(num_class) + '.pkl'
    else:
        label_encoder_path = 'input/label_encoder_' + str(num_class) + '.pkl'
    # seeding
    # TODO refactor seeding. free some seed
    np.random.seed(seed)
    random.seed(seed + 1)
    set_random_seed(seed + 2)
    raw_ds_tr = RawDataSet(data_path=raw_meta_train_path)
    results = []
    output_path = None
    now = datetime.datetime.now()
    cur_time = now.strftime('%m_%d_%H_%M')
    out_file_name = cur_time + '_' + protocol
    if protocol in ['val', 'cv', 'cv_val']:
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
        valid_iter = raw_ds_tr.split_dev_val_iter(seed=seed + 5, num_repeat=num_repeat)
    elif protocol == 'cv_submit':
        valid_iter = raw_ds_tr.cv_iter(seed=seed + 5, num_repeat=num_repeat, yield_test_set=False)
        valid_iter = ((tr, raw_ds_te) for tr in valid_iter)
    elif protocol == 'cv_val':
        valid_iter = raw_ds_tr.split_dev_val_iter(seed=seed + 5, num_repeat=num_repeat)
    else:
        valid_iter = raw_ds_tr.cv_iter(seed=seed + 5, num_repeat=num_repeat)
        # TODO eval blending predictions performance on cv
    for i, (raw_ds_tr, raw_ds_te) in enumerate(valid_iter):
        if protocol not in ['cv_submit', 'cv_val']:
            ds_tr = VQADataSet(raw_ds_tr, feature_path=train_resource_path, label_encoder_path=label_encoder_path,
                               multi_label=data_opts.multi_label, len_q=data_opts.len_q, num_class=data_opts.num_class,
                               frame_aggregate_strategy=data_opts.frame_aggregate_strategy,
                               len_video=data_opts.len_video,
                               batch_size=data_opts.batch_size, lazy_load=data_opts.lazy_load, seed=seed + 6 + i)

            model = model_func(ds_tr, **model_params)
            model.fit_generator(ds_tr, epochs=epochs, )
            ds_tr.clear()
            ds_te = VQADataSet(raw_ds_te, multi_label=data_opts.multi_label, len_q=data_opts.len_q,
                               num_class=data_opts.num_class,
                               batch_size=data_opts.test_batch_size, is_test=True, shuffle_data=False,
                               feature_path=test_resource_path, label_encoder_path=label_encoder_path,
                               frame_aggregate_strategy=data_opts.frame_aggregate_strategy, lazy_load=data_opts.lazy_load)
            p_te = model.predict_generator(ds_te)
            ds_te.clear()
            if artifact_dir is not None:
                output_path_raw_prediction = os.path.join(artifact_dir, out_file_name + '.npy')
                output_path_ds = os.path.join(output_dir, out_file_name + '_ds.pkl')
                save(p_te, output_path_raw_prediction)
                save(ds_te, output_path_ds)
            score = ds_te.eval_or_submit(p_te, output_path=output_path)
            results.append(score)
        else:
            predictions = []
            ds_te = VQADataSet(raw_ds_te, multi_label=data_opts.multi_label, len_q=data_opts.len_q,
                               num_class=data_opts.num_class,
                               batch_size=data_opts.test_batch_size, is_test=True, shuffle_data=False,
                               feature_path=test_resource_path, label_encoder_path=label_encoder_path,
                               frame_aggregate_strategy=data_opts.frame_aggregate_strategy,
                               lazy_load=data_opts.lazy_load)
            for raw_ds_dev in raw_ds_tr.cv_iter(seed=seed + 233, num_repeat=10, yield_test_set=False):
                ds_tr = VQADataSet(raw_ds_dev, feature_path=train_resource_path, label_encoder_path=label_encoder_path,
                                   multi_label=data_opts.multi_label, len_q=data_opts.len_q,
                                   num_class=data_opts.num_class,
                                   frame_aggregate_strategy=data_opts.frame_aggregate_strategy,
                                   len_video=data_opts.len_video,
                                   batch_size=data_opts.batch_size, lazy_load=data_opts.lazy_load, seed=seed + 6 + i)

                model = model_func(ds_tr, **model_params)
                model.fit_generator(ds_tr, epochs=epochs, )
                ds_tr.clear()
                p_te = model.predict_generator(ds_te)
                predictions.append(p_te)

            predictions_total = np.zeros_like(predictions[0])
            for p in predictions:
                predictions_total += p
            predictions_total /= len(predictions)
            results = ds_te.eval_or_submit(predictions_total, output_path=output_path)
            ds_te.clear()
            if artifact_dir is not None:
                # TODO handle artifact saving in cv
                output_path_raw_prediction = os.path.join(artifact_dir, out_file_name + '.npy')
                output_path_ds = os.path.join(output_dir, out_file_name + '_ds.pkl')
                save(predictions_total, output_path_raw_prediction)
                save(ds_te, output_path_ds)
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
            new_results = []
            for r in results:
                new_results.append('%.3f' % r)
            results = new_results

    return results
