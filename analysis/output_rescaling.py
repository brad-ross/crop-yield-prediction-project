import tensorflow as tf
import numpy as np
import pandas as pd
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

VAL_YEAR = 2013
states_to_keep = np.array([5, 17, 18, 19, 20, 27, 29, 31, 38, 39, 46])

soy_data = np.load(os.path.expanduser('~/cs231n-satellite-images-hist/data_soybean_filtered.npz'))
hist_sums = np.sum(soy_data['output_image'],axis=(1,2,3))
nonbroken_rows = hist_sums > 287
imp_rows = pd.DataFrame(soy_data['output_index'])[0].isin(states_to_keep)
val_year_rows = soy_data['output_year'] == VAL_YEAR
index_validate = np.logical_and.reduce((nonbroken_rows, imp_rows, val_year_rows))

soy_yield = soy_data['output_yield'][index_validate]

corn_data = np.load(os.path.expanduser('~/cs231n-satellite-images-hist/data_corn.npz'))
corn_yield = corn_data['output_yield'][index_validate]

# Scaling Soy Predictions to Corn
corn_preds_w_soy_mod = predict_with_model(corn_data['output_image'][index_validate], os.path.expanduser('~/models/run1__dropout_0.50__soybean'), '2013CNN_model.ckpt.meta')
std_corn_preds = (corn_preds_w_soy_mod - np.mean(soy_yield))/np.std(soy_yield)
unstd_corn_preds = std_corn_preds * np.std(corn_yield) + np.mean(corn_yield)
corn_rmse = np.sqrt(np.mean((unstd_corn_preds - corn_yield)**2))
print(corn_rmse)

# Scaling Corn Predictions to Soy
soy_preds_w_corn_mod = predict_with_model(soy_data['output_image'][index_validate], os.path.expanduser('~/models/run2__dropout_0.50__corn'), 'important_counties2013CNN_model.ckpt.meta')
std_soy_preds = (soy_preds_w_corn_mod - np.mean(corn_yield))/np.std(corn_yield)
unstd_soy_preds = std_soy_preds * np.std(soy_yield) + np.mean(soy_yield)
soy_rmse = np.sqrt(np.mean((unstd_soy_preds - soy_yield)**2))
print(soy_rmse)

# writing out rescaled predictions
np.savez(os.path.expanduser('~/cs231n-satellite-images-models/output_rescaling/original_model_rescaled_oututs.npz'), soy_preds=soy_preds_w_corn_mod, corn_preds=corn_preds_w_soy_mod, locs=soy_data['output_index'][index_validate])
