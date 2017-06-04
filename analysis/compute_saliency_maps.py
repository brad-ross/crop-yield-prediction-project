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
    y = graph.get_tensor_by_name('Placeholder:0')
    keep_prob = graph.get_tensor_by_name('Placeholder_2:0')
    loss = graph.get_tensor_by_name('L2Loss:0')

    return x, y, keep_prob, loss

def compute_saliency_maps_for(hists, labels, model_path, meta_graph):
    with tf.Session() as sess:
        graph = load_model(sess, model_path, meta_graph)
        x, y, keep_prob, loss = get_relevant_nodes_from(graph)
        hist_grad = tf.gradients(loss, [x])[0]
        
        feed_dict = {x: hists, y: labels, keep_prob: 1}
        loss, hist_grad_val = sess.run([loss, hist_grad], feed_dict=feed_dict)
        rmse = np.sqrt(loss*2/hists.shape[0])
        norm_hist_grad = hist_grad_val / np.amax(np.absolute(hist_grad_val), axis=(1, 2, 3)).reshape((-1, 1, 1, 1))
        return norm_hist_grad, rmse

VAL_YEAR = 2013
states_to_keep = np.array([5, 17, 18, 19, 20, 27, 29, 31, 38, 39, 46])

# Soybean Saliency Maps
soy_data = np.load(os.path.expanduser('~/cs231n-satellite-images-hist/data_soybean_filtered.npz'))

hist_sums = np.sum(soy_data['output_image'],axis=(1,2,3))
nonbroken_rows = hist_sums > 287
imp_rows = pd.DataFrame(soy_data['output_index'])[0].isin(states_to_keep)
val_year_rows = soy_data['output_year'] == VAL_YEAR
index_validate = np.logical_and.reduce((nonbroken_rows, imp_rows, val_year_rows))

soy_model_path = os.path.expanduser('~/models/run1__dropout_0.50__soybean')
soy_sal_maps, soy_rmse = compute_saliency_maps_for(soy_data['output_image'][index_validate], soy_data['output_yield'][index_validate], soy_model_path, '2013CNN_model.ckpt.meta')
print(soy_sal_maps.shape, soy_rmse)

# Corn Saliency Maps
corn_data = np.load(os.path.expanduser('~/cs231n-satellite-images-hist/data_corn.npz'))

corn_model_path = os.path.expanduser('~/models/run2__dropout_0.50__corn')
corn_sal_maps, corn_rmse = compute_saliency_maps_for(corn_data['output_image'][index_validate], corn_data['output_yield'][index_validate], corn_model_path, 'important_counties2013CNN_model.ckpt.meta')
print(corn_sal_maps.shape, corn_rmse)

# Saving Saliency Maps
output_path = os.path.expanduser('~/cs231n-satellite-images-models/saliency_maps/original_model_comparison_imp_count.npz')
np.savez(output_path, soy_maps=soy_sal_maps, corn_maps=corn_sal_maps, index=soy_data['output_index'][index_validate])
