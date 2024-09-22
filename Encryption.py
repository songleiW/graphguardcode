import sys
import time
import numpy as np
import Query_Process
import fileIO
import mpc
MAX = 2147483647 #sys.maxsize
import random
import math
import syft as sy
from Crypto.Random import get_random_bytes
from numba import cuda


def encGraph(G,NL):
    startIDs1=np.random.randint(MAX,size=len(G),dtype=np.int32)
    startIDs2=np.random.randint(MAX,size=len(G),dtype=np.int32)
    startIDs3=G[:,0]-startIDs1-startIDs2

    endIDs1 = np.random.randint(MAX,size=len(G),dtype=np.int32)
    endIDs2 = np.random.randint(MAX,size=len(G),dtype=np.int32)
    endIDs3 = G[:,1]-endIDs1-endIDs2

    label_one_hots1 = np.random.randint(2,size=(len(G),NL), dtype=np.bool)
    label_one_hots2 = np.random.randint(2,size=(len(G),NL), dtype=np.bool)

    label_one_hots3 = np.zeros((len(G),NL), dtype=np.bool)

    for i in range(len(G)):
        label_one_hots3[i][G[i][2]]=1
        label_one_hots3[i]=label_one_hots3[i]^label_one_hots1[i]^label_one_hots2[i]




    eGraph1=[]
    eGraph2=[]
    eGraph3=[]

    eGraph1.append(startIDs1)
    eGraph1.append(endIDs1)
    eGraph1.append(label_one_hots1)
    eGraph1.append(G[:,3])

    eGraph2.append(startIDs2)
    eGraph2.append(endIDs2)
    eGraph2.append(label_one_hots2)

    eGraph3.append(startIDs3)
    eGraph3.append(endIDs3)
    eGraph3.append(label_one_hots3)



    return eGraph1,eGraph2,eGraph3



def encQuery(Query, Order,NL):
    startIDs1 = np.random.randint(MAX, size=len(Query), dtype=np.int32)
    startIDs2 = np.random.randint(MAX, size=len(Query), dtype=np.int32)
    startIDs3 = Query[:, 0] - startIDs1 - startIDs2

    endIDs1 = np.random.randint(MAX, size=len(Query), dtype=np.int32)
    endIDs2 = np.random.randint(MAX, size=len(Query), dtype=np.int32)
    endIDs3 = Query[:, 1] - endIDs1 - endIDs2

    label_one_hots1 = np.random.randint(2, size=(len(Query), NL), dtype=np.bool)
    label_one_hots2 = np.random.randint(2, size=(len(Query), NL), dtype=np.bool)
    label_one_hots3 = np.zeros((len(Query), NL), dtype=np.bool)

    for i in range(len(Query)):
        label_one_hots3[i][Query[i][2]]=1
        label_one_hots3[i]=label_one_hots3[i]^label_one_hots1[i]^label_one_hots2[i]

    eorder11=np.random.randint(2, size=(len(Query), len(Query)), dtype=np.bool)
    eorder21=np.random.randint(2, size=(len(Query), len(Query)), dtype=np.bool)

    eorder12=np.random.randint(2, size=(len(Query), len(Query)), dtype=np.bool)
    eorder22=np.random.randint(2, size=(len(Query), len(Query)), dtype=np.bool)

    eorder13=eorder11^eorder12
    eorder23=eorder21^eorder22

    for i in range(len(Query)):
        for j in range(len(Query)):
            if Order[i][j]==1:
                eorder13[i][j]=eorder13[i][j]^True
            elif Order[i][j]==2:
                eorder23[i][j]=eorder23[i][j]^True

    eQuery1 = []
    eQuery2 = []
    eQuery3 = []

    eQuery1.append(startIDs1)
    eQuery1.append(endIDs1)
    eQuery1.append(label_one_hots1)
    eQuery1.append(eorder11)
    eQuery1.append(eorder21)


    eQuery2.append(startIDs2)
    eQuery2.append(endIDs2)
    eQuery2.append(label_one_hots2)
    eQuery2.append(eorder12)
    eQuery2.append(eorder22)


    eQuery3.append(startIDs3)
    eQuery3.append(endIDs3)
    eQuery3.append(label_one_hots3)
    eQuery3.append(eorder13)
    eQuery3.append(eorder23)


    return eQuery1, eQuery2, eQuery3

# Define the number of parties
n_parties = 3

# Define the shape of the streaming graph
streaming_graph_shape = (10, 10)

# Define the number of possible values for each private value
n_values = 5

# Generate a random key for the encryption
key = get_random_bytes(16)

# Define the encoding function to convert a value to its one-hot encoding
def encode(value):
    one_hot = np.zeros(n_values)
    one_hot[value] = 1
    return one_hot

# Define the streaming graph values (assumed to be integers between 0 and n_values-1)
streaming_graph_values = np.random.randint(0, n_values, streaming_graph_shape)

# Encrypt the streaming graph using the replicated secret sharing technique
streaming_graph_encrypted = []
for party_id in range(n_parties):
    party_encryption = []
    for i in range(streaming_graph_shape[0]):
        row_encryption = []
        for j in range(streaming_graph_shape[1]):
            value = streaming_graph_values[i][j]
            one_hot_encoding = encode(value)
            party_share = np.random.randint(0, 2, n_values)
            shares = [party_share] + [np.random.randint(0, 2, n_values) for _ in range(n_parties-1)]
            row_encryption.append(shares)
        party_encryption.append(row_encryption)
    streaming_graph_encrypted.append(party_encryption)

# Print the encrypted streaming graph for each party
for party_id in range(n_parties):
    print(f"Encrypted streaming graph for party {party_id}:")
    print(np.array(streaming_graph_encrypted[party_id]))



def one_hot_encode(value, num_classes) :
    """
    Converts a given value into its one-hot encoding form.

    Args:
    value (int): The value to be converted into one-hot encoding.
    num_classes (int): The total number of classes in the dataset.

    Returns:
    numpy.ndarray: A one-dimensional NumPy array representing the one-hot encoding of the given value.
    """
    one_hot = np.zeros(num_classes)
    one_hot[value] = 1
    return one_hot