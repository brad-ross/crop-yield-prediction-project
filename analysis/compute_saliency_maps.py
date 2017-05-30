import tensorflow as tf
import numpy as np
import os

MODEL_PATH = '~/models/run1__dropout-0.50'

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
        norm_hist_grad = hist_grad_val / np.amax(hist_grad_val, axis=(1, 2, 3)).reshape((-1, 1, 1, 1))
        return norm_hist_grad, rmse

def load_imgs_and_labels(data_path):
    data = np.load(data_path)
    return data['output_image'], data['output_yield'], data['output_locations']

# Soybean Saliency Maps
soy_data = np.load(os.path.expanduser('~/cs231n-satellite-images-hist/data_soybean.npz'))
index_validate = np.nonzero(soy_data['output_year'] == 2013)[0]

soy_sal_maps, soy_rmse = compute_saliency_maps_for(soy_data['output_image'][index_validate], soy_data['output_yield'][index_validate], os.path.expanduser('~/models/run0__dropout-0.25'), '2013CNN_model.ckpt.meta')
print(soy_sal_maps.shape, soy_rmse)

# Corn Saliency Maps
corn_data = np.load(os.path.expanduser('~/cs231n-satellite-images-hist/data_corn.npz'))
index_validate = np.nonzero(corn_data['output_year'] == 2013)[0]

corn_sal_maps, corn_rmse = compute_saliency_maps_for(corn_data['output_image'][index_validate], corn_data['output_yield'][index_validate], os.path.expanduser('~/models/run3__dropout-0.25__corn'), '2013CNN_model.ckpt.meta')
print(corn_sal_maps.shape, corn_rmse)
