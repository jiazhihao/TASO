import argparse
import tensorflow as tf
import numpy as np
import time
from shared_functions import make_activation, make_conv2d

def resnet_block(input, strides, out_channels, name):
    t = make_conv2d(input_tensor=input, filter_shape=(1,1,input.shape[1].value,out_channels), strides=(1,1,1,1), padding="SAME", actimode="RELU", name=name+"_conv1")
    t = make_conv2d(input_tensor=t, filter_shape=(3,3,out_channels,out_channels), strides=strides, padding="SAME", actimode="RELU", name=name+"_conv2")
    t = make_conv2d(input_tensor=t, filter_shape=(1,1,out_channels,out_channels*4), strides=(1,1,1,1), padding="SAME", actimode="NONE", name=name+"_conv3")
    if (strides[2]>1) or (input.shape[1].value != out_channels * 4):
        input = make_conv2d(input_tensor=input, filter_shape=(1,1,input.shape[1].value,out_channels*4), strides=strides, padding="SAME", actimode="RELU", name=name+"_conv4")
    return tf.nn.relu(tf.add(input, t))

parser = argparse.ArgumentParser()
parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=1000)
parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1000)
args = parser.parse_args()

input = tf.placeholder(tf.float32, shape=(1,64,56,56))
t = input
for i in range(3):
    t = resnet_block(t, (1,1,1,1), 64, "resnet_block_1_{}".format(i))
strides=(1,1,2,2)
for i in range(4):
    t = resnet_block(t, strides, 128, "resnet_block_2_{}".format(i))
    strides=(1,1,1,1)
strides=(1,1,2,2)
for i in range(6):
    t = resnet_block(t, strides, 256, "resnet_block_3_{}".format(i))
    strides=(1,1,1,1)
strides=(1,1,2,2)
for i in range(3):
    t = resnet_block(t, strides, 512, "resnet_block_4_{}".format(i))
    strides=(1,1,1,1)

config = tf.ConfigProto()
if (args.xla):
    print("Measuring inference performance with XLA ON")
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
else:
    print("Measuring inference performance with XLA OFF")
print(config.graph_options.optimizer_options.global_jit_level)

output_nodes = [t]
input_dictionary = {}
input_dictionary[input] = np.random.random_sample((1,64,56,56))

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
