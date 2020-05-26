#https://blog.csdn.net/qq_36793545/article/details/84844867

import torch
import numpy as np

#图的邻接矩阵表示
#Adjacency matrix representation of directed graph
A = np.array([
[0, 1, 0, 0],
[0, 0, 1, 1],
[0, 1, 0, 0],
[1, 0, 1, 0]],
dtype=float
)

#节点的特征
#Features of each node
X = np.array([
[i, -i] for i in range(A.shape[0])
], dtype=float)

#传播规则 A* X
#每个节点的特征均由邻居节点的特征表示
#Propagation rules A* X
#The representation of each node (each row) is now the sum of its neighbor features. 
#In other words, the graph convolution layer characterizes each node with its neighbors
p1 = np.dot(A, X)

#问题1
#节点特征表示不包含本身的特征
#Question 1
#A nodes representation do not contain their own features. 
#The representation of a node is the collection of the characteristics of its neighbors, 
#so only self-loop nodes will contain their own characteristics after the collection
I = np.eye(A.shape[0])
A += I
p2 = np.dot(A, X)

#问题2
#拥有大度数的节点在特征表征中会有较大的值，
#而度较小的节点特征表征的值也会小
#这会导致梯度消失或者梯度爆炸
#Question2
#Nodes with a large input degree will have a larger value in the feature representation, 
#while nodes with a smaller input degree will have a smaller value. 
#This will cause the gradient vanishing or the gradient exploding
D = np.diag(np.sum(A, axis=0))
A = np.dot(np.linalg.inv(D), A)
p3 = np.dot(A, X)

# Add weights
W = np.array([
[1, -1],
[-1, 1]
])
p4 = np.dot(np.dot(A, X), W)
print ('Before relu:\n', p4)

#Reduce the demination to reduce the output demination
W1 = np.array([
[1],
[-1]
])
p5 = np.dot(np.dot(A, X), W1)

def relu(x):
    return np.maximum(0, x)

# Add activate fucntion ReLU
print ('Before relu:\n', relu(p4))

# A REAL CASE
# Used for Graph data
import networkx as nx
from networkx import to_numpy_matrix
import matplotlib.pyplot as plt

def plot_graph(G, weight_name=None):
    '''
    G: a networkx G
    weight_name: name of the attribute for plotting edge weights (if G is weighted)
    '''
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = None
    
    if weight_name:
        weights = [int(G[u][v][weight_name]) for u,v in edges]
        labels = nx.get_edge_attributes(G,weight_name)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        nx.draw_networkx(G, pos, edges=edges, width=weights);
    else:
        nodelist1 = []
        nodelist2 = []
        for i in range (34):
            if zkc.nodes[i]['club'] == 'Mr. Hi':
                nodelist1.append(i)
            else:
                nodelist2.append(i)
        #nx.draw_networkx(G, pos, edges=edges);
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist1, node_size=300, node_color='r',alpha = 0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist2, node_size=300, node_color='b',alpha = 0.8)
        nx.draw_networkx_edges(G, pos, edgelist=edges,alpha =0.4)

#空手道俱乐部数据集
zkc = nx.karate_club_graph()
#plot_graph(zkc)
#plt.show()

order = sorted(list(zkc.nodes()))
A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())
A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

W_1 = np.random.normal(loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
W_2 = np.random.normal(loc=0, size=(W_1.shape[1], 2))

def gcn_layer(A_hat, D_hat, X, W):
    return relu(D_hat ** -1 * A_hat * X * W)

H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)
output = H_2

feature_representations = {
    node: np.array(output)[node]
    for node in zkc.nodes()}

feature_representations

for i in range (34):
    if zkc.nodes[i]['club'] == 'Mr. Hi':
        plt.scatter(np.array(output)[i,0],np.array(output)[i,1] ,color = 'b',alpha=0.5,s = 100)
    else:
        plt.scatter(np.array(output)[i,0],np.array(output)[i,1] ,color = 'r',alpha=0.5,s = 100)
#plt.scatter(np.array(output)[:,0],np.array(output)[:,1])
plt.show()
