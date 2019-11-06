import argparse
import taso as ts
import onnx

NUM_LAYERS = 12

def create_layer_weights(graph, num_layers, channels):
    """Instantiates weights for each layer.

       Args:
        graph: The TASO graph.
        num_layers: The number of layers.
        channels: The number of channels.

       Returns:
        A list of lists of weights (one list for each layer).
    """
    all_w = []
    for i in range(num_layers):
        w = []
        # conv_3x3
        w.append(graph.new_weight(dims=(channels, channels, 3, 3))) # conv_3x3
        # conv_5x5
        w.append(graph.new_weight(dims=(channels, channels, 5, 5))) # conv_5x5
        # separable conv_3x3
        w.append(graph.new_weight(dims=(channels, 1, 3, 3)))
        w.append(graph.new_weight(dims=(channels, 1, 3, 3)))
        # separable conv_5x5
        w.append(graph.new_weight(dims=(channels, 1, 5, 5)))
        w.append(graph.new_weight(dims=(channels, 1, 5, 5)))

        all_w.append(w)
    return all_w

def get_dims(t):
    """Returns the size of a tensor."""
    return tuple([t.dim(i) for i in range(4)])

def add_skips(graph, layers, skips):
    """Adds the output from the specified skip connections.

       Computes the input to the next layer by summing together the outputs
       from the specified skip connections with the output from the previous
       layer. Downsamples outputs with appropriately sized average pooling
       layers to ensure that all tensors have the same size before summing.

       Args:
        graph: The TASO graph.
        layers: A list of the output tensors of each layer.
        skips: A list where element i is 1 if the current layer
               has a skip connection to layer i and 0 otherwise.

       Returns:
        The sum of all skip connections and the output of the previous layer.
    """
    t = layers[-1]
    sizes = \
        set([get_dims(layers[i]) for i in range(len(skips)) if skips[i] == 1])
    sizes.add(get_dims(t))
    min_size = min(sizes)
    for i in range(len(skips)):
        if skips[i] == 1:
            size = get_dims(layers[i])
            if size > min_size:
                kernel_size = size[-1] - 2 * (min_size[-1] - 1)
                t = graph.add(
                        graph.avgpool2d(input=layers[i],
                                        kernels=[kernel_size, kernel_size],
                                        strides=[2, 2],
                                        padding="VALID"),
                        t)
            else:
                t = graph.add(layers[i], t)
    return t

def separable_conv(graph, input, all_w, kernel_dim, layer_id):
    """Defines a separable convolution.

       Args:
        graph: The TASO graph.
        input: The input tensor.
        all_w: A list of lists of weights (one list for each layer).
        kernel_dim: The size of the kernel.
        layer_id: The ID of the layer for which the separable conv is generated.

       Returns:
        The output of the separable conv.
    """
    if kernel_dim == 3:
        conv_w = all_w[layer_id][2:4]
    elif kernel_dim == 5:
        conv_w = all_w[layer_id][4:]
    else:
        raise ValueError('Invalid kernel dim for '
                         'separable conv: %d' % (kernel_dim))
    t = graph.conv2d(input=input, weight=conv_w[0], strides=(1, 1),
                     padding="SAME", activation="RELU")
    return graph.conv2d(input=t, weight=conv_w[1], strides=(1, 1),
                            padding="SAME", activation="RELU")

def create_architecture(arc, graph, input, all_w):
    """Creates an architecture with shared weights.

       Instantiates a new architecture using the specified string
       representation. Each layer is one of six operators: conv_3x3,
       separable conv_3x3, conv_5x5, separable conv_5x5, average poooling,
       or max pooling (see <https://github.com/melodyguan/enas> for details).
       Each stateful layer is initialized with pre-defined weights which will be
       shared with parallel architectures.

       Args:
        arc: The string representation of the architecture.
        graph: The TASO graph.
        input: The input tensor.
        all_w: A list of lists of weights (one list for each layer).
    """
    t = input
    layers = arc.split('|')

    y = []
    for i, layer in enumerate(layers):
        spec = [int(x) for x in layer.split(' ')]
        if i > 0:
            t = add_skips(graph, y, spec[1:])
        if spec[0] == 0:
           t = graph.conv2d(input=t, weight=all_w[i][0], strides=(1, 1),
                            padding="SAME", activation="RELU")
        elif spec[0] == 1:
           t = separable_conv(graph, t, all_w, kernel_dim=3, layer_id=i)
        elif spec[0] == 2:
           t = graph.conv2d(input=t, weight=all_w[i][1], strides=(1, 1),
                            padding="SAME", activation="RELU")
        elif spec[0] == 3:
           t = separable_conv(graph, t, all_w, kernel_dim=5, layer_id=i)
        elif spec[0] == 4:
           t = graph.avgpool2d(input=t, kernels=[3, 3], strides=[2, 2],
                               padding="SAME", activation="NONE")
        elif spec[0] == 5:
           t = graph.maxpool2d(input=t, kernels=[3, 3], strides=[2, 2],
                               padding="SAME", activation="NONE")
        y.append(t)

def parse_arcs(input_file):
    """Extracts the architecture string representations from an input file."""
    arcs = []
    with open(input_file, 'r') as f:
        for line in f:
            arcs.append(line.strip())
    return arcs

def main(args):
    graph = ts.new_graph()
    input_size = tuple([int(x) for x in args.input_size.split('x')])
    input = graph.new_input(dims=input_size)
    all_w = create_layer_weights(graph, NUM_LAYERS, args.channels)
    all_arcs = parse_arcs(args.input_file)
    if args.num_models is not None:
        all_arcs = all_arcs[:args.num_models]

    # stem conv
    t = graph.conv2d(input=input,
                     weight=graph.new_weight(dims=(args.channels,
                                                   input.dim(1), 1, 1)),
                     strides=(1, 1), padding="SAME", activation="RELU")

    for arc in all_arcs:
        create_architecture(arc, graph, t, all_w)

    if args.save_models:
        onnx_model = ts.export_onnx(graph)
        onnx.save(onnx_model, 'original_model.onnx')

    new_graph = ts.optimize(graph, alpha=1.0, budget=1000)
    if args.save_models:
        onnx_model = ts.export_onnx(new_graph)
        onnx.save(onnx_model, 'optimized_model.onnx')

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Runs architectures sampled by the ENAS algorithm in TASO')
    parser.add_argument('--input_file', type=str,
                        default='examples/enas_arcs.txt',
                        help='Input file specifying ENAS models')
    parser.add_argument('--num_models', type=int, default=None,
                        help=('Number of models to fuse '
                              '(fuses all models if not specified)'))
    parser.add_argument('--input_size', type=str, default='1x3x32x32',
                        help='Input size ("NxCxHxW")')
    parser.add_argument('--channels', type=int, default=32,
                        help='Number of channels')
    parser.add_argument('--save_models', action='store_true', default=False,
                        help=('If set, saves original and optimized models in '
                              'ONNX form'))
    args = parser.parse_args()
    main(args)
