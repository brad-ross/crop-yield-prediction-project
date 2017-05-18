import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(filename, plot_title, max_val, x_scale):
    result = np.load(filename)
    train_loss = np.fmin(result['summary_train_loss'], max_val)
    eval_loss = np.fmin(result['summary_eval_loss'], max_val)
    max_x = min(train_loss.shape[0], eval_loss.shape[0])
    x = np.array(range(max_x)) * x_scale
    t_line, = plt.plot(x, train_loss, label='Training Loss')
    v_line, = plt.plot(x, eval_loss, label='Validation Loss')
    plt.legend((t_line, v_line), ('Training Loss', 'Validation Loss'))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(plot_title)
    plt.show()
    
if len(sys.argv) <= 1:
    print("Usage: python plot_loss.py [filename] [title [max_val [x_scale]?]?]?")
else:
    filename = sys.argv[1]
    plot_title = sys.argv[2] if len(sys.argv) > 2 else 'Training and Validation Loss'
    max_val = int(sys.argv[3]) if len(sys.argv) > 3 else 5000.0
    x_scale = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    plot_loss(filename, plot_title, max_val, x_scale)

                                                                            
