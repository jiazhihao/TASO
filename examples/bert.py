import taso as ts

seq_length = 64
hidden_dims = 1024
batch_size = 16

def attention(graph, input, heads):
    d_model = input.dim(1)
    d_k = d_model // heads
    assert input.dim(1) % heads == 0
    weights = list()
    for i in range(3):
        weights.append(graph.new_weight(dims=(d_model, d_model)))
    # compute query, key, value tensors
    q = graph.matmul(input, weights[0])
    k = graph.matmul(input, weights[1])
    v = graph.matmul(input, weights[2])
    # reshape query, key, value to multiple heads
    q = graph.reshape(q, shape=(batch_size, 64,16,64))
    k = graph.reshape(k, shape=(batch_size, 64,16,64))
    v = graph.reshape(v, shape=(batch_size, 64,16,64))
    # transpose query, key, value for batched matmul
    q = graph.transpose(q, perm=(0,2,1,3), shuffle=True)
    k = graph.transpose(k, perm=(0,2,1,3), shuffle=True)
    v = graph.transpose(v, perm=(0,2,1,3), shuffle=True)
    # perform matrix multiplications
    logits = graph.matmul(q, k)
    output = graph.matmul(logits, v)
    # transpose the output back
    output = graph.transpose(output,perm=(0,2,1,3), shuffle=True)
    output = graph.reshape(output, shape=(batch_size * 64, 1024))

    # a final linear layer
    linear = graph.new_weight(dims=(d_model, d_model))
    output = graph.matmul(output, linear)
    return output

graph = ts.new_graph()
input = graph.new_input(dims=(batch_size * seq_length, hidden_dims))
input = graph.relu(input)
t = input
for i in range(8):
    t = attention(graph, t, 16)

new_graph = ts.optimize(graph, alpha=1.0, budget=100)
