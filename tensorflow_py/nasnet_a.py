import argparse
import tensorflow as tf
import numpy as np
import time
from shared_functions import make_activation, make_conv2d, make_seperable_conv2d, make_avgpool2d, make_maxpool2d

def squeeze(out_channels, input):
    return make_conv2d(input_tensor=input, filter_shape=(1,1,input.shape[1].value,out_channels), strides=(1,1,1,1), padding="SAME", actimode="RELU", name="squeeze")

def fit(current, input):
    if (input.shape[2].value == current.shape[2].value):
        return squeeze(current.shape[1].value, input)
    else:
        return make_conv2d(input_tensor=input, filter_shape=(3,3,input.shape[1].value,current.shape[1].value), strides=(1,1,2,2), padding="SAME", actimode="RELU", name="fit")

def normal_cell(prev, cur, out_channels):
    cur = squeeze(out_channels, cur)
    prev = fit(cur, prev)
    ts = list()
    ts.append(make_seperable_conv2d(input_tensor=cur, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(cur)
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=cur, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_avgpool2d(input_tensor=cur, kernels=(1,1,3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(prev)
    ts.append(make_avgpool2d(input_tensor=prev, kernels=(1,1,3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_avgpool2d(input_tensor=prev, kernels=(1,1,3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    assert len(ts) == 10
    outputs=list()
    for i in range(5):
        outputs.append(tf.add(ts[2*i], ts[2*i+1]))
    return tf.concat(outputs, axis=1, name="concat1")

def reduction_cell(prev, cur, out_channels):
    cur = squeeze(out_channels, cur)
    prev = fit(cur, prev)
    ts = list()
    outputs = list()
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(7,7), strides=(1,1,2,2), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=cur, out_channels=out_channels, kernels=(5,5), strides=(1,1,2,2), padding="SAME"))
    outputs.append(tf.add(ts[0], ts[1]))
    ts.append(make_maxpool2d(input_tensor=cur, kernels=(1,1,3,3), strides=(1,1,2,2), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(7,7), strides=(1,1,2,2), padding="SAME"))
    outputs.append(tf.add(ts[2], ts[3]))
    ts.append(make_avgpool2d(input_tensor=cur, kernels=(1,1,3,3), strides=(1,1,2,2), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=prev, out_channels=out_channels, kernels=(5,5), strides=(1,1,2,2), padding="SAME"))
    outputs.append(tf.add(ts[4], ts[5]))
    ts.append(make_maxpool2d(input_tensor=cur, kernels=(1,1,3,3), strides=(1,1,2,2), padding="SAME"))
    ts.append(make_seperable_conv2d(input_tensor=outputs[0], out_channels=out_channels, kernels=(3,3), strides=(1,1,1,1), padding="SAME"))
    outputs.append(tf.add(ts[6], ts[7]))
    ts.append(make_avgpool2d(input_tensor=outputs[0], kernels=(1,1,3,3), strides=(1,1,1,1), padding="SAME"))
    ts.append(outputs[1])
    outputs.append(tf.add(ts[8], ts[9]))
    return tf.concat(outputs, axis=1, name="concat2")

parser = argparse.ArgumentParser()
parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=1000)
parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1000)
args = parser.parse_args()

input0 = tf.placeholder(tf.float32, shape=(1,128,56,56))
input = input0
out_channels = 128
for i in range(3):
    if i > 0:
        input = reduction_cell(prev, cur, out_channels)
    prev = input
    cur = input
    for j in range(10):
        t = normal_cell(prev, cur, out_channels)
        prev = cur
        cur = t
    out_channels *= 2

config = tf.ConfigProto()
if (args.xla):
    print("Measuring inference performance with XLA ON")
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
else:
    print("Measuring inference performance with XLA OFF")
print(config.graph_options.optimizer_options.global_jit_level)

output_nodes = [t]
input_dictionary = {}
input_dictionary[input0] = np.random.random_sample((1,128,56,56))

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
