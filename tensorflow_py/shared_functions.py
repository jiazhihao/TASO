import tensorflow as tf
import numpy as np

def make_activation(input, actimode, name):
    if actimode == "NONE":
        return input
    elif actimode == "RELU":
        relu_name = name + "_relu"
        relu = tf.nn.relu(input, name=relu_name)
        return relu
    elif actimode == "SIGMOID":
        sigmoid_name = name + "_sigmoid"
        sigmoid = tf.nn.sigmoid(input, name=sigmoid_name)
        return sigmoid
    elif actimode == "TANH":
        tanh_name = name + "_tanh"
        tanh = tf.nn.tanh(input, name=tanh_name)
        return tanh
    else:
        print("Unknown Actimode")
        assert(0)

def make_conv2d(input_tensor, filter_shape, strides, padding, actimode, name):
    weights_name = name + "_weights"
    conv_name = name + "_conv2d"
    weights = tf.constant(np.random.random_sample(filter_shape), name=weights_name, dtype=tf.float32)
    conv2d = tf.nn.conv2d(input_tensor, weights, strides, padding, data_format="NCHW", name=conv_name)
    return make_activation(conv2d, actimode, name)

def make_seperable_conv2d(input_tensor, out_channels, kernels, strides, padding, actimode="NONE", name="seperable_conv2d"):
    depthwise_filter_shape=(kernels[0],kernels[1],input_tensor.shape[1].value,1)
    pointwise_filter_shape=(1,1,input_tensor.shape[1].value,out_channels)
    dp_filter = tf.constant(np.random.random_sample(depthwise_filter_shape), name=name+"_dp_filter", dtype=tf.float32)
    pw_filter = tf.constant(np.random.random_sample(pointwise_filter_shape), name=name+"_pw_filter", dtype=tf.float32)
    conv2d = tf.nn.separable_conv2d(input_tensor, dp_filter, pw_filter, strides, padding, data_format="NCHW", name=name)
    return make_activation(conv2d, actimode, name)

def make_avgpool2d(input_tensor, kernels, strides, padding):
    return tf.nn.avg_pool(input_tensor, kernels, strides, padding, data_format="NCHW")

def make_maxpool2d(input_tensor, kernels, strides, padding):
    return tf.nn.max_pool(input_tensor, kernels, strides, padding, data_format="NCHW")

def make_matmul(input_tensor, out_channels):
    weight_shape = (input_tensor.shape[1].value, out_channels)
    weight = tf.constant(np.random.random_sample(weight_shape), dtype=tf.float32)
    return tf.matmul(input_tensor, weight)
