import argparse
import tensorflow as tf
import numpy as np
import time
from shared_functions import make_matmul

def attention(input, heads):
    d_model = input.shape[1].value
    q = make_matmul(input, d_model)
    k = make_matmul(input, d_model)
    v = make_matmul(input, d_model)
    # reshape query, key, value
    q = tf.reshape(q, shape=(64,16,64))
    k = tf.reshape(k, shape=(64,16,64))
    v = tf.reshape(v, shape=(64,16,64))
    # transpose q, k, v for batched matmul
    q = tf.transpose(q, perm=(1,0,2))
    k = tf.transpose(k, perm=(1,0,2))
    v = tf.transpose(v, perm=(1,0,2))
    logits = tf.matmul(q, k)
    output = tf.matmul(logits, v)
    # transpose the output back
    output = tf.transpose(output, perm=(1,0,2))
    output = tf.reshape(output, shape=(64, 1024))
    # a final linear layer
    output = make_matmul(tf.nn.relu(make_matmul(input, 4*d_model)), d_model)
    return output

parser = argparse.ArgumentParser()
parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=1000)
parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1000)
args = parser.parse_args()

input = tf.placeholder(tf.float32, shape=(64,1024))
input_dictionary = {}
input_dictionary[input] = np.random.random_sample((64, 1024))
t = input
for i in range(12):
    t = attention(t, 16)

output_nodes = [t]

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
