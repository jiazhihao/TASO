import taso

def get_pads(kernel, padding):
    if sum(padding) == 0 and sum(kernel) > 2:
        pads = "VALID"
    else:
        pads = "SAME"
    return pads

def conv2d(graph, v, out_channels, kernel=(1, 1), stride=(1, 1), padding=(0, 0)):
    w = graph.new_weight(dims=(out_channels, v.dim(1), *kernel))
    v = graph.conv2d(input=v, weight=w, strides=stride, padding=get_pads(kernel, padding), activation="RELU")
    return v

def pool2d(graph, v, pool_type, kernel=(1, 1), stride=(1, 1), padding=(0, 0)):
    if pool_type == 'global_avg':
        pads = "VALID"
        x = graph.avgpool2d(input=v, kernels=kernel, strides=[1, 1], padding=pads)
    elif pool_type == 'avg':
        pads = get_pads(kernel, padding)
        x = graph.avgpool2d(input=v, kernels=kernel, strides=stride, padding=pads)
    elif pool_type == 'max':
        pads = get_pads(kernel, padding)
        x = graph.maxpool2d(input=v, kernels=kernel, strides=stride, padding=pads)
    else:
        raise NotImplemented
    return x
        

def inception_front(graph, v):  # 3 x 299 x 299
    v = conv2d(graph, v, out_channels=32, kernel=(3, 3), stride=(2, 2))  # 32 x 149 x 149
    v = conv2d(graph, v, out_channels=32, kernel=(3, 3))  # 32 x 147 x 147
    v = conv2d(graph, v, out_channels=64, kernel=(3, 3), padding=(1, 1))  # 64 x 147 x 147
    v = pool2d(graph, v, pool_type='max', kernel=(3, 3), stride=(2, 2))  # 64 x 73 x 73
    v = conv2d(graph, v, 80, kernel=(1, 1))  # 80 x 73 x 73
    v = conv2d(graph, v, out_channels=192, kernel=(3, 3))  # 192 x 71 x 71
    v = pool2d(graph, v, pool_type='max', kernel=(3, 3), stride=(2, 2))  # 192 x 35 x 35
    return v


def inception_a(graph, v, pool_features):
    v1x1 = conv2d(graph, v, out_channels=64, kernel=(1, 1))

    v5x5 = conv2d(graph, v, out_channels=48, kernel=(1, 1))
    v5x5 = conv2d(graph, v5x5, out_channels=64, kernel=(5, 5), padding=(2, 2))

    v3x3dbl = conv2d(graph, v, out_channels=64, kernel=(1, 1))
    v3x3dbl = conv2d(graph, v3x3dbl, out_channels=96, kernel=(3, 3), padding=(1, 1))
    v3x3dbl = conv2d(graph, v3x3dbl, out_channels=96, kernel=(3, 3), padding=(1, 1))

    v_pool = pool2d(graph, v, pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    v_pool = conv2d(graph, v_pool, out_channels=pool_features, kernel=(1, 1))
    return graph.concat(1, [v1x1, v5x5, v3x3dbl, v_pool])


def inception_b(graph, v):
    v3x3 = conv2d(graph, v, out_channels=384, kernel=(3, 3), stride=(2, 2))

    v3x3dbl = conv2d(graph, v, out_channels=64, kernel=(1, 1))
    v3x3dbl = conv2d(graph, v3x3dbl, out_channels=96, kernel=(3, 3), padding=(1, 1))
    v3x3dbl = conv2d(graph, v3x3dbl, out_channels=96, kernel=(3, 3), stride=(2, 2))

    v_pool = pool2d(graph, v, pool_type='max', kernel=(3, 3), stride=(2, 2))
    return graph.concat(1, [v3x3, v3x3dbl, v_pool]);


def inception_c(graph, v, channels_7x7):
    v1x1 = conv2d(graph, v, out_channels=192, kernel=(1, 1))

    c7 = channels_7x7
    v7x7 = conv2d(graph, v, out_channels=c7, kernel=(1, 1))
    v7x7 = conv2d(graph, v7x7, out_channels=c7, kernel=(1, 7), padding=(0, 3))
    v7x7 = conv2d(graph, v7x7, out_channels=192, kernel=(7, 1), padding=(3, 0))

    v7x7dbl = conv2d(graph, v, out_channels=c7, kernel=(1, 1))
    v7x7dbl = conv2d(graph, v7x7dbl, out_channels=c7, kernel=(7, 1), padding=(3, 0))
    v7x7dbl = conv2d(graph, v7x7dbl, out_channels=c7, kernel=(1, 7), padding=(0, 3))
    v7x7dbl = conv2d(graph, v7x7dbl, out_channels=c7, kernel=(7, 1), padding=(3, 0))
    v7x7dbl = conv2d(graph, v7x7dbl, out_channels=192, kernel=(1, 7), padding=(0, 3))

    v_pool = pool2d(graph, v, pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    v_pool = conv2d(graph, v_pool, out_channels=192, kernel=(1, 1))
    return graph.concat(1, [v1x1, v7x7, v7x7dbl, v_pool])


def inception_d(graph, v):
    v3x3 = conv2d(graph, v, out_channels=192, kernel=(1, 1))
    v3x3 = conv2d(graph, v3x3, out_channels=320, kernel=(3, 3), stride=(2, 2))

    v7x7x3 = conv2d(graph, v, out_channels=192, kernel=(1, 1))
    v7x7x3 = conv2d(graph, v7x7x3, out_channels=192, kernel=(1, 7), padding=(0, 3))
    v7x7x3 = conv2d(graph, v7x7x3, out_channels=192, kernel=(7, 1), padding=(3, 0))
    v7x7x3 = conv2d(graph, v7x7x3, out_channels=192, kernel=(3, 3), stride=(2, 2))

    v_pool = pool2d(graph, v, pool_type='max', kernel=(3, 3), stride=(2, 2))
    return graph.concat(1, [v3x3, v7x7x3, v_pool])


def inception_e(graph, v):
    v1x1 = conv2d(graph, v, out_channels=320, kernel=(1, 1))

    v3x3 = conv2d(graph, v, out_channels=384, kernel=(1, 1))
    v3x3a = conv2d(graph, v3x3, out_channels=384, kernel=(1, 3), padding=(0, 1))
    v3x3b = conv2d(graph, v3x3, out_channels=384, kernel=(3, 1), padding=(1, 0))

    v3x3dbl = conv2d(graph, v, out_channels=448, kernel=(1, 1))
    v3x3dbl = conv2d(graph, v3x3dbl, out_channels=384, kernel=(3, 3), padding=(1, 1))
    v3x3dbla = conv2d(graph, v3x3dbl, out_channels=384, kernel=(1, 3), padding=(0, 1))
    v3x3dblb = conv2d(graph, v3x3dbl, out_channels=384, kernel=(3, 1), padding=(1, 0))

    v_pool = pool2d(graph, v, pool_type='avg', kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    v_pool = conv2d(graph, v_pool, out_channels=192, kernel=(1, 1))
    return graph.concat(1, [v1x1, v3x3a, v3x3b, v3x3dbla, v3x3dblb, v_pool])


def inception_logits(graph, v):
    return pool2d(graph, v, pool_type='global_avg')


def inception_v3(batch_size=1):
    graph = taso.new_graph()
    v = graph.new_input(dims=(batch_size, 3, 299, 299))
    v = inception_front(graph, v)
    v = inception_a(graph, v, 32)
    v = inception_a(graph, v, 64)
    v = inception_a(graph, v, 64)
    v = inception_b(graph, v)
    v = inception_c(graph, v, 128)
    v = inception_c(graph, v, 160)
    v = inception_c(graph, v, 160)
    v = inception_c(graph, v, 192)
    v = inception_d(graph, v)
    v = inception_e(graph, v)
    v = inception_e(graph, v)
    v = inception_logits(graph, v)
    return graph

graph = inception_v3(batch_size=32)  # change batch_size from 4 to 8 would cause error.
opt_graph = taso.optimize(graph, alpha=1.0, budget=30)

print(graph.run_time())
print(opt_graph.run_time())
