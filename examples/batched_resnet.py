import argparse
import onnx

import taso as ts

def resnet_block(graph, input, strides, out_channels, shared_w=None):
    if shared_w is not None:
        w1 = shared_w[0]
        w2 = shared_w[1]
        w3 = shared_w[2]
        w4 = shared_w[3]
    else:
        w1 = None
        w2 = None
        w3 = None
        w4 = None

    if w1 is None:
        w1 = graph.new_weight(dims=(out_channels,input.dim(1),1,1))
    t = graph.conv2d(input=input, weight=w1,
                     strides=(1,1), padding="SAME",
                     activation="RELU")
    if w2 is None:
        w2 = graph.new_weight(dims=(out_channels,t.dim(1),3,3))
    t = graph.conv2d(input=t, weight=w2,
                     strides=strides, padding="SAME",
                     activation="RELU")
    if w3 is None:
        w3 = graph.new_weight(dims=(4*out_channels,t.dim(1),1,1))
    t = graph.conv2d(input=t, weight=w3,
                     strides=(1,1), padding="SAME")
    if (strides[0]>1) or (input.dim(1) != out_channels*4):
        if w4 is None:
            w4 = graph.new_weight(dims=(out_channels*4,input.dim(1),1,1))
        input=graph.conv2d(input=input, weight=w4,
                           strides=strides, padding="SAME",
                           activation="RELU")
    return (graph.relu(graph.add(input, t)), [w1, w2, w3, w4])

def resnet_model(graph, input, all_w=None, num_shared_blocks=None):
    if all_w is None:
        use_shared_w = False
        all_w = []
    else:
        use_shared_w = True
    t = input
    j = 0
    for i in range(3):
        if use_shared_w and j < num_shared_blocks:
            t, weights = resnet_block(graph, t, (1,1), 64,
                                      shared_w=all_w[j])
        else:
            t, weights = resnet_block(graph, t, (1,1), 64)
            all_w.append(weights)
        j += 1
    strides = (2,2)
    for i in range(4):
        if use_shared_w and j < num_shared_blocks:
            t, weights = resnet_block(graph, t, strides, 128,
                                      shared_w=all_w[j])
        else:
            t, weights = resnet_block(graph, t, strides, 128)
            all_w.append(weights)
        j += 1
        strides = (1,1)
    return all_w

def shared_resnet_model(graph, input, num_models, num_shared_blocks):
    all_w = resnet_model(graph, input)
    for i in range(1, num_models):
        resnet_model(graph, input, all_w, num_shared_blocks)
    return graph

def main(args):
    graph = ts.new_graph()
    input_size = tuple([int(x) for x in args.input_size.split('x')])
    input = graph.new_input(dims=input_size)
    shared_resnet_model(graph, input, args.num_models, args.num_shared_blocks)
    if args.save_graphs:
        original_model = ts.export_onnx(graph)
        onnx.save(original_model, 'original_model.onnx')

    new_graph = ts.optimize(graph, alpha=1.0, budget=1000)
    if args.save_graphs:
        optimized_model = ts.export_onnx(new_graph)
        onnx.save(optimized_model, 'optimized_model.onnx')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--num_models', type=int, default=1,
                        help='Number of parallel models')
    parser.add_argument('--num_shared_blocks', type=int, default=0,
                        help='Number of shared blocks')
    parser.add_argument('--input_size', type=str, default='1x3x32x32',
                        help='Input size ("NxCxHxW")')
    parser.add_argument('--save_graphs', action='store_true', default=False,
                        help=('If set, saves original and optimized models in '
                              'ONNX form'))
    args = parser.parse_args()
    main(args)
