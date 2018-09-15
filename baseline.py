# baseline : use vgg16 4096 feature for each frame
import pandas as pd

from  data import VQASequence, ResetCallBack
from model import get_baseline_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
vqa_tr = VQASequence(batch_size=128)
vqa_te = VQASequence(meta_data_path='input/test.txt', feature_path='output_te.h5', is_test=True, shuffle_data=False,
                     batch_size=128)
model = get_baseline_model(vqa_tr)
model.fit_generator(vqa_tr, epochs=20, callbacks=[ResetCallBack(vqa_tr)])
p_te = model.predict_generator(vqa_te)
data_to_submit = vqa_te.create_submit(p_te)
df = pd.DataFrame(data_to_submit)
df.to_csv('submit.txt', index=False, header=None, encoding='utf-8')
