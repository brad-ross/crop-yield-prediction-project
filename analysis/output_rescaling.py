import tensorflow as tf
import numpy as np
import os

def load_model(sess, model_path, meta_graph):
    saver = tf.train.import_meta_graph(os.path.join(model_path, meta_graph))
    saver.restore(sess, tf.train.latest_checkpoint(os.path.expanduser(model_path)))
    return tf.get_default_graph()

def get_relevant_nodes_from(graph):
    x = graph.get_tensor_by_name('x:0')
    keep_prob = graph.get_tensor_by_name('Placeholder_2:0')
    logits = graph.get_tensor_by_name('Squeeze:0')

    return x, keep_prob, logits

def predict_with_model(hists, model_path, meta_graph):
    with tf.Session() as sess:
        graph = load_model(sess, model_path, meta_graph)
        x, keep_prob, logits = get_relevant_nodes_from(graph)

        feed_dict = {x: hists, keep_prob: 1}
        return sess.run(logits, feed_dict=feed_dict)

soy_data = np.load(os.path.expanduser('~/cs231n-satellite-images-hist/data_soybean_filtered.npz'))
index_validate = np.nonzero(soy_data['output_year'] == 2013)[0]
soy_yield = soy_data['output_yield'][index_validate]

corn_data = np.load(os.path.expanduser('~/cs231n-satellite-images-hist/data_corn.npz'))
index_validate = np.nonzero(corn_data['output_year'] == 2013)[0]
corn_yield = corn_data['output_yield'][index_validate]

# Scaling Soy Predictions to Corn
corn_preds_w_soy_mod = predict_with_model(corn_data['output_image'][index_validate], os.path.expanduser('~/models/run0__dropout-0.25'), '2013CNN_model.ckpt.meta')
std_corn_preds = (corn_preds_w_soy_mod - np.mean(soy_yield))/np.std(soy_yield)
unstd_corn_preds = std_corn_preds * np.std(corn_yield) + np.mean(corn_yield)
corn_rmse = np.sqrt(np.mean((unstd_corn_preds - corn_yield)**2))
print(corn_rmse)

# Scaling Corn Predictions to Soy
soy_preds_w_corn_mod = predict_with_model(soy_data['output_image'][index_validate], os.path.expanduser('~/models/run3__dropout-0.25__corn'), '2013CNN_model.ckpt.meta')
std_soy_preds = (soy_preds_w_corn_mod - np.mean(corn_yield))/np.std(corn_yield)
unstd_soy_preds = std_soy_preds * np.std(soy_yield) + np.mean(soy_yield)
soy_rmse = np.sqrt(np.mean((unstd_soy_preds - soy_yield)**2))
print(soy_rmse)
