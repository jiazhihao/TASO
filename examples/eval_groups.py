import xflow

graph = xflow.new_graph()
input = graph.new_input(dims=(1,512,28,28))
input = graph.maxpool2d(input=input, kernels=(1,1), strides=(1,1), padding="SAME")
# Printing the performance of different multi-branch convolutions
graph.print_measurements()
i = 1
while i <= 32:
    print("Num. Convs Per Grop = {}".format(i))
    weight = graph.new_weight(dims=(512,512//i,3,3))
    t = graph.conv2d(input=input,weight=weight,strides=(1,1),padding="SAME", activation="RELU")
    i *= 2

#weight1 = graph.new_weight(dims=(256,8,3,3))
#t1 = graph.conv2d(input=input,weight=weight1,strides=(1,1), padding="SAME", activation="RELU")
#weight2 = graph.new_weight(dims=(256,16,3,3))
#t2 = graph.conv2d(input=input,weight=weight2,strides=(1,1), padding="SAME", activation="RELU")
#weight3 = graph.new_weight(dims=(256,32,3,3))
#t3 = graph.conv2d(input=input,weight=weight3,strides=(1,1), padding="SAME", activation="RELU")
#weight4 = graph.new_weight(dims=(256,64,3,3))
#t4 = graph.conv2d(input=input,weight=weight4,strides=(1,1), padding="SAME", activation="RELU")
#weight5 = graph.new_weight(dims=(256,128,3,3))
#t5 = graph.conv2d(input=input,weight=weight5,strides=(1,1), padding="SAME", activation="RELU")
#weight6 = graph.new_weight(dims=(256,256,3,3))
#t6 = graph.conv2d(input=input,weight=weight6,strides=(1,1), padding="SAME", activation="RELU")

