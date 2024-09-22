import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import syft as sy
import networkx as nx
from numba import cuda
epsilon=1.6
delta=0.00001

MAX = 2147483647 #sys.maxsize


def RSS_ADD(x1, x2, x3, y1, y2, y3) :
    return x1 + y1, x2 + y2, x3 + y3


def RSS_MUL(x1, x2, x3, y1, y2, y3) :
    return x1 * y1 + x1 * y2 + x2 * y1, x2 * y2 + x2 * y3 + x3 * y2, x3 * y3 + x3 * y1 + x1 * y3

def ASS_MUL(x1, y1, x2, y2) :


    u = np.random.randint(0, math.sqrt(MAX))

    v = np.random.randint(0, math.sqrt(MAX))

    w = u*v
    u1=np.random.randint(0, MAX)
    u2 = u - u1
    v1 =np.random.randint(0, MAX)
    v2 = v - v1
    w1 = np.random.randint(0, MAX)
    w2 = w - w1
    e = x1 - u1 + x2 - u2
    f = y1 - v1 + y2 - v2
    res1 = e * f + f * u1 + e * v1 + w1
    res2 = f * u2 + e * v2 + w2
    return res1, res2

def Shuffle(PI12, PI23, PI31, a1, a2, a3) :
    T12 = np.random.randint(MAX, size=a3.shape)
    T23 = np.random.randint(MAX, size=a3.shape)
    T31 = np.random.randint(MAX, size=a3.shape)
    R1 = np.random.randint(MAX, size=a3.shape)
    R2 = np.random.randint(MAX, size=a3.shape)
    X = np.zeros(a3.shape, dtype=int)
    Y = np.zeros(a3.shape, dtype=int)
    C1 = np.zeros(a3.shape, dtype=int)
    C2 = np.zeros(a3.shape, dtype=int)
    R3 = np.zeros(a3.shape, dtype=int)

    X[PI12] = a1 + a2 + T12
    X[PI31] = X + T31

    Y[PI12] = a3 - T12
    C1[PI23] = X + T23
    C1 = C1 - R2

    C2[PI31] = Y - T31
    C2[PI23] = C2 - T23
    C2 = C2 - R1

    R3 = C1 + C2

    return R1, R2, R3


def unShuffle(PI12, PI23, PI31, a1, a2, a3) :
    T12 = np.random.randint(MAX, size=a3.shape)
    T23 = np.random.randint(MAX, size=a3.shape)
    T31 = np.random.randint(MAX, size=a3.shape)
    R1 = np.random.randint(MAX, size=a3.shape)
    R2 = np.random.randint(MAX, size=a3.shape)
    X = np.zeros(a3.shape, dtype=int)
    Y = np.zeros(a3.shape, dtype=int)
    C1 = np.zeros(a3.shape, dtype=int)
    C2 = np.zeros(a3.shape, dtype=int)
    R3 = np.zeros(a3.shape, dtype=int)

    X = (a1 + a2 + T12)[PI23]
    X = (X + T31)[PI31]

    Y = (a3 - T12)[PI23]
    C1 = (X + T23)[PI12]
    C1 = C1 - R2

    C2 = (Y - T31)[PI31]
    C2 = (C2 - T23)[PI12]
    C2 = C2 - R1

    R3 = C1 + C2

    print(R1 + R2 + R3)

    return R1, R2, R3


SIZE=10
vals=np.random.randint(1000,size=SIZE)
keys=np.random.choice(SIZE, SIZE, replace=False)

vals1=np.random.randint(MAX,size=SIZE)
vals2=np.random.randint(MAX,size=SIZE)
vals3=vals-vals1-vals2

keys1=np.random.randint(MAX,size=SIZE)
keys2=np.random.randint(MAX,size=SIZE)
keys3=keys-keys1-keys2

PI1=np.random.choice(SIZE, SIZE, replace=False)
PI2=np.random.choice(SIZE, SIZE, replace=False)
PI3=np.random.choice(SIZE, SIZE, replace=False)




print(vals1+vals2+vals3)
R1,R2,R3=mpc.Shuffle(PI1,PI2,PI3,vals1,vals2,vals3)
mpc.unShuffle(PI1,PI2,PI3,R1,R2,R3)


import numpy as np
import syft as sy

class ReplicatedSecretSharing:
    def __init__(self, n_parties):
        self.n_parties = n_parties
        self.hook = sy.TorchHook(torch)  # initialize PySyft hook
        self.parties = [sy.VirtualWorker(self.hook, id=f"worker_{i}") for i in range(self.n_parties)]
        self.crypto_provider = sy.VirtualWorker(self.hook, id='crypto_provider')  # create a virtual worker for crypto functions

    def share_onehot(self, onehot):
        # convert the one-hot encoding to an integer value and share it among the parties
        value = np.argmax(onehot)
        value_ptrs = sy.MultiPointerTensor([torch.tensor(value)], shape=(1, self.n_parties)).share(*self.parties, crypto_provider=self.crypto_provider)
        return value_ptrs

    def equal(self, value_ptrs1, value_ptrs2):
        # perform the equality test using the encrypted values
        result_ptrs = value_ptrs1 == value_ptrs2
        return result_ptrs

    def reveal(self, value_ptrs):
        # retrieve the result of the equality test from the parties and decrypt it
        result = value_ptrs.get().sum().item()
        return result

# example usage
n_parties = 3

# initialize the replicated secret sharing scheme
rss = ReplicatedSecretSharing(n_parties)

# generate the one-hot encoded values to be compared
value1 = np.array([0, 1, 0, 0])
value2 = np.array([0, 0, 1, 0])

# share the one-hot encoded values among the parties
value_ptrs1 = rss.share_onehot(value1)
value_ptrs2 = rss.share_onehot(value2)

# perform the equality test on the encrypted values
result_ptrs = rss.equal(value_ptrs1, value_ptrs2)

# retrieve the result of the equality test from the parties and decrypt it
result = rss.reveal(result_ptrs)



class ReplicatedSecretSharing:
    def __init__(self, n_parties):
        self.n_parties = n_parties
        self.hook = sy.TorchHook(torch)  # initialize PySyft hook
        self.parties = [sy.VirtualWorker(self.hook, id=f"worker_{i}") for i in range(self.n_parties)]
        self.crypto_provider = sy.VirtualWorker(self.hook, id='crypto_provider')  # create a virtual worker for crypto functions

    def share_matrix(self, matrix):
        # share the matrix among the parties using the replicated secret sharing technique
        matrix_ptrs = sy.MultiPointerTensor([torch.tensor(matrix)], shape=(matrix.shape[0], matrix.shape[1], self.n_parties)).share(*self.parties, crypto_provider=self.crypto_provider)
        return matrix_ptrs

    def combine_matrices(self, matrix_ptrs_list):
        # calculate the combination of the adjacency matrices using the encrypted matrices
        combined_matrix_ptrs = sum(matrix_ptrs_list)
        return combined_matrix_ptrs

    def isomorphic(self, matrix_ptrs1, matrix_ptrs2):
        # check whether the resulting adjacency matrix is isomorphic with the given adjacency matrix
        matrix1 = matrix_ptrs1.get().sum(dim=2)
        matrix2 = matrix_ptrs2.get().sum(dim=2)
        G1 = nx.from_numpy_array(matrix1.cpu().numpy())
        G2 = nx.from_numpy_array(matrix2.cpu().numpy())
        result = nx.is_isomorphic(G1, G2)
        return result

    def reveal(self, value_ptrs):
        # retrieve the result from the parties and decrypt it
        result = value_ptrs.get().sum().item()
        return result

# example usage
n_parties = 3

# initialize the replicated secret sharing scheme
rss = ReplicatedSecretSharing(n_parties)

# generate the adjacency matrices to be combined
matrix1 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
matrix2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
matrix3 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

# share the adjacency matrices among the parties
matrix_ptrs1 = rss.share_matrix(matrix1)
matrix_ptrs2 = rss.share_matrix(matrix2)
matrix_ptrs3 = rss.share_matrix(matrix3)

# calculate the combination of the adjacency matrices
combined_matrix_ptrs = rss.combine_matrices([matrix_ptrs1, matrix_ptrs2, matrix_ptrs3])

# generate the given adjacency matrix
query_matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])

# check whether the resulting adjacency matrix is isomorphic with the given adjacency matrix
result_ptrs = rss.isomorphic(combined_matrix_ptrs, rss.share_matrix(query_matrix))






def draw_Laplace_noise(Delta,epsilon,delta):



    mu = -1 * Delta * (math.log((np.exp(epsilon / Delta) + 1) * (1 - np.power(1-delta, 1 / Delta)))) / epsilon
    #print(mu)
    mu = -1 * Delta * (math.log((np.exp(epsilon / Delta) + 1) * (1 - np.sqrt(1 - delta)))) / epsilon

    print(mu)
    return
    a=(np.exp(epsilon/Delta-1))/(np.exp(epsilon/Delta+1))
    print(a)

    for i in range(0,9000,50):
        #print(i/100)
        print(a*np.exp(-epsilon*np.abs(i-mu)/Delta))



    exit(11)

    mu = -1 * Delta * (math.log((np.exp(epsilon / Delta) + 1) * (1 - np.sqrt(1 - delta)))) / epsilon

    mu=Delta*10
    for i in range(100):
        print(np.int32(np.random.laplace(mu, Delta / epsilon)))

    a=9200
    b=100
    mu=a/2

    x=range(0,a,b)
    y=np.zeros(np.int32(a/b))

    for i in range(np.int32(a/b)):
        y[i]=(np.exp(epsilon / Delta) - 1)/(np.exp(epsilon / Delta) + 1)*np.exp(-epsilon*np.abs(x[i]-mu)/Delta)

    plt.plot(x, y)

    plt.show()


    exit(11)

    noiseList = []

    p = (np.exp(epsilon / Delta) - 1) / (np.exp(epsilon / Delta) + 1)
    scale = p * np.exp(-epsilon * (np.abs(min - mu)) / Delta)
    for i in range(min, max + 1) :
        n = round(p * np.exp(-epsilon * (np.abs(i - mu)) / Delta) / scale)
        for j in range(n) :
            noiseList.append(i)


    return noiseList[np.random.randint(len(noiseList))]


def B2A(B1, B2, B3):

    R=np.random.randint(MAX, size=len(B1), dtype=np.int32)
    M0=(False^B1^B3).astype(np.int32)-R
    M1=(True^B1^B3).astype(np.int32)-R

    Mb=np.logical_not(B2).astype(np.int32)*M0+B2.astype(np.int32)*M1

    return R,Mb,np.zeros(B1.shape,dtype=np.int32)






