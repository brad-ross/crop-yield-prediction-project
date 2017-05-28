import tensorflow as tf
import os

MODEL_PATH = '~/models/run1__dropout-0.50'

sess = tf.Session()

saver = tf.train.import_meta_graph(os.path.expanduser(os.path.join(MODEL_PATH, '2013CNN_model.ckpt.meta')))
saver.restore(sess, tf.train.latest_checkpoint(os.path.expanduser(MODEL_PATH)))



gvars = tf.global_variables()
gvars_vals = sess.run(gvars)
