# Main source: http://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
# Another source: http://stackoverflow.com/questions/38511166/while-debugging-how-to-print-all-variables-which-is-in-list-format-who-are-tr

import tensorflow as tf
import os

sess = tf.Session()
saver = tf.train.import_meta_graph(os.path.expanduser('~/cs231n-satellite-images-models/2013CNN_model.ckpt.meta'))
saver.restore(sess, tf.train.latest_checkpoint(os.path.expanduser('~/cs231n-satellite-images-models/')))
gvars = tf.global_variables()
gvars_vals = sess.run(gvars)
print("List of variables:")
for var in gvars:
    print(var.name)
print()
vars_to_print = ['conv1_1/conv2d/W:0', 'conv1_1/conv2d/b:0']
for var, var_val in zip(gvars, gvars_vals):
    if var.name in vars_to_print:
        print("Value of %s:" % var.name)
        print(var_val)
        print()
