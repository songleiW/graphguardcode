import numpy as np
import syft as sy
from numba import cuda


def readGraph(flag) :
    if flag == 1 :
        fileName = "datasets/mooc_actions/mooc_actions.tsv"  # the number of labels in mooc_actions dataset is 97
        N = 411749  # the number of edges in the dataset
    elif flag == 2 :
        fileName = "datasets/Hyperlinks/Hyperlinks.tsv"  # the number of labels in mooc_actions dataset is 97
        N = 858488  # the number of edges in the dataset
    else:
        fileName = "datasets/DBLP/DBLP.txt"  # the number of labels in mooc_actions dataset is 97
        N = 1049866  # the number of edges in the dataset

    file = open(fileName)
    G = np.zeros((N, 4), dtype=np.int32)  # startID,EndID, Label, timestamp

    line = file.readline().strip()
    num = 0
    while line :
        index = line.split('\t')
        G[num] = np.array(index).astype(np.int32)
        num = num + 1
        line = file.readline()
    print("Graph data read success!")
    return G

def processHyperlinks():
    writefileName = "datasets/Hyperlinks/Hyperlinks.tsv"
    writeFile = open(writefileName, "w")
    mappings = {}

    for i in range(2):
        if i==0:
            readfileName = "datasets/Hyperlinks/soc-redditHyperlinks-body.tsv"  # the number of labels in mooc_actions dataset is 97
        elif i==1:
            readfileName = "datasets/Hyperlinks/soc-redditHyperlinks-title.tsv"  # the number of labels in mooc_actions dataset is 97
        else:
            readfileName = "datasets/DBLP/DBLP.txt"


        readfFile = open(readfileName)
        line = readfFile.readline().strip()
        while line:
            index = line.split('\t')
            if index[0] in mappings:
                index[0]=mappings[index[0]]
            else:
                mappings[index[0]]=len(mappings)
                index[0]=len(mappings)-1

            if index[1] in mappings:
                index[1]=mappings[index[1]]
            else:
                mappings[index[1]]=len(mappings)
                index[1]=len(mappings)-1

            if index[3]=="-1":
                index[3]=0

            writeFile.write(str(index[0])+"\t"+str(index[1])+"\t"+str(index[3])+"\t"+str(index[2])+"\n")
            line = readfFile.readline().strip()

        readfFile.close()

    writeFile.close()

    exit(11)



def readQuery(flag, query_size) :
    if flag == 1 :
        queryName = "datasets/mooc_actions/query_" + str(query_size) + "_1.tsv"
        timeName = "datasets/mooc_actions/query_" + str(query_size) + "_1_time.tsv"

    elif flag == 2 :
        queryName = "datasets/Hyperlinks/query_" + str(query_size) + "_1.tsv"
        timeName = "datasets/Hyperlinks/query_" + str(query_size) + "_1_time.tsv"
    else:
        queryName = "datasets/DBLP/query_" + str(query_size) + "_1.tsv"
        timeName = "datasets/DBLP/query_" + str(query_size) + "_1_time.tsv"

    # -----------------read the query graph-----------------#
    file = open(queryName)
    Q = np.zeros((query_size, 3), dtype=np.int32)  # startID,EndID, Label
    size_Qc = [0]
    line = file.readline().strip()
    num = 0
    while line :
        if line == "\n" :
            size_Qc.append(num)  # remember the size of the TC-graph
            line = file.readline()
            continue

        index = line.split('\t')
        Q[num] = np.array(index).astype(np.int32)
        num = num + 1
        line = file.readline()
    size_Qc.append(num)
    file.close()

    # -----------------read the timing constraints-----------------#
    file = open(timeName)
    Order = np.zeros((query_size, query_size), dtype=np.int32)
    line = file.readline().strip()
    num = 0
    while line :
        index = line.split('\t')
        Order[num] = np.array(index).astype(np.int32)
        num = num + 1
        line = file.readline().strip()

    print("Query read success!")

    return Q, Order, size_Qc



class StreamingGraph :
    def __init__(self, n_nodes, n_parties) :
        self.n_nodes = n_nodes
        self.n_parties = n_parties
        self.hook = sy.TorchHook(torch)  # initialize PySyft hook
        self.parties = [sy.VirtualWorker(self.hook, id=f"worker_{i}") for i in range(self.n_parties)]
        self.crypto_provider = sy.VirtualWorker(self.hook,
                                                id='crypto_provider')  # create a virtual worker for crypto functions

        # generate the initial adjacency matrix
        self.adj_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        self.adj_ptrs = sy.MultiPointerTensor([torch.from_numpy(self.adj_matrix)],
                                              shape=(1, n_parties, n_nodes, n_nodes)).share(*self.parties,
                                                                                            crypto_provider=self.crypto_provider)

    def add_edge(self, node1, node2) :
        # update the adjacency matrix
        self.adj_matrix[node1][node2] = 1
        self.adj_matrix[node2][node1] = 1

        # share the updated adjacency matrix among the parties
        self.adj_ptrs = sy.MultiPointerTensor([torch.from_numpy(self.adj_matrix)],
                                              shape=(1, self.n_parties, self.n_nodes, self.n_nodes)).share(
            *self.parties, crypto_provider=self.crypto_provider)

    def get_adj_matrix(self) :
        # retrieve the final adjacency matrix from the parties and decrypt it
        adj_matrix = self.adj_ptrs.get()[0].get()
        return adj_matrix


# example usage
n_nodes = 5
n_parties = 3

# initialize the streaming graph
graph = StreamingGraph(n_nodes, n_parties)

# add some edges to the graph
graph.add_edge(0, 1)
graph.add_edge(1, 2)
graph.add_edge(2, 3)

# get the final adjacency matrix
adj_matrix = graph.get_adj_matrix()
