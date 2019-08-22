import argparse
import tensorflow as tf
import numpy as np
import time
from shared_functions import make_matmul

hidden_size = 512
length = 5

def combine(x, h):
    w1 = make_matmul(x, hidden_size)
    w2 = make_matmul(h, hidden_size)
    return tf.add(tf.nn.relu(w1), tf.nn.relu(w2))

def nas_node(input, x):
    t = list()
    for i in range(8):
        t.append(combine(x, input))
    midt = list()
    midt.append(tf.add(tf.nn.relu(t[0]), tf.nn.sigmoid(t[3])))
    midt.append(tf.add(tf.nn.sigmoid(t[1]), tf.nn.tanh(t[2])))
    midt.append(tf.multiply(tf.nn.sigmoid(t[4]), tf.nn.tanh(t[5])))
    midt.append(tf.multiply(tf.nn.sigmoid(t[6]), tf.nn.relu(t[7])))
    midt.append(tf.add(tf.nn.sigmoid(midt[1]), tf.nn.tanh(midt[2])))
    midt.append(tf.multiply(tf.nn.tanh(midt[0]), tf.nn.tanh(midt[3])))
    midt.append(tf.multiply(tf.nn.tanh(midt[4]), tf.nn.tanh(midt[5])))
    return tf.nn.tanh(midt[6])

parser = argparse.ArgumentParser()
parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=1000)
parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1000)
args = parser.parse_args()

input_dictionary = {}
xs = list()
output_nodes = []
for i in range(length):
    xs.append(tf.placeholder(tf.float32, shape=(1, hidden_size)))
    input_dictionary[xs[i]] = np.random.random_sample((1, hidden_size))
state = tf.constant(np.random.random_sample((1, hidden_size)), dtype=tf.float32)
for i in range(length):
    state = nas_node(state, xs[i])
    output_nodes.append(state)

config = tf.ConfigProto()
if (args.xla):
    print("Measuring inference performance with XLA ON")
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
else:
    print("Measuring inference performance with XLA OFF")
print(config.graph_options.optimizer_options.global_jit_level)

with tf.Session(config=config) as sess:
    if (args.print_tensorboard):
        writer = tf.summary.FileWriter(args.print_tensorboard, sess.graph)
    times = []
    for i in range(args.discard_iter + args.iterations):
        t0 = time.time()
        sess.run(output_nodes, input_dictionary)
        t1 = time.time()
        times.append(t1 - t0)
    total = 0
    for i in range(args.discard_iter, len(times)):
        total += times[i]
    avg = total / (args.iterations) * 1000.0
    print("Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")
