# baseline : use vgg16 4096 feature for each frame
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef

from  data import VQASequence, ResetCallBack
from model import get_baseline_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
vqa_va = VQASequence(meta_data_path='input/val.txt', shuffle_data=False,
                     batch_size=256)
dummy_predictions = np.ones((vqa_va.answers.shape[0], 1)) * np.argmax(np.sum(vqa_va.answers, axis=0))
print(accuracy_score(np.argmax(vqa_va.answers, axis=1), dummy_predictions))
vqa_tr = VQASequence(meta_data_path='input/dev.txt', batch_size=128)

model = get_baseline_model(vqa_tr)
model.fit_generator(vqa_tr, epochs=20, callbacks=[ResetCallBack(vqa_tr)])
p_te = model.predict_generator(vqa_va)

print(accuracy_score(np.argmax(vqa_va.answers, axis=1), np.argmax(p_te, axis=1)))
print(matthews_corrcoef(np.argmax(vqa_va.answers, axis=1), np.argmax(p_te, axis=1)))
