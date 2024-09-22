import math
import time
import numpy as np
import mpc
import FSS
MAX = 2147483647 #sys.maxsize
import syft as sy
from numba import cuda

def secPattern_Detection(eGraph1,eGraph2,eGraph3, WINDOW_SIZE, eQuery1,eQuery2,eQuery3, size_Qc,NL):
    for t in range(WINDOW_SIZE, len(eGraph1[0])):
        print(t)
        time_start = time.time()
        MatchsTimeList1=[]
        MatchsTimeList2=[]
        MatchsTimeList3=[]

        MatchsIDList1=[]
        MatchsIDList2 = []
        MatchsIDList3 = []

        eGt1 = []
        eGt2 = []
        eGt3 = []
        eGt1.append(eGraph1[0][t - WINDOW_SIZE :t])
        eGt1.append(eGraph1[1][t - WINDOW_SIZE :t])
        eGt1.append(eGraph1[2][t - WINDOW_SIZE :t,:])
        eGt1.append(eGraph1[3][t - WINDOW_SIZE :t])

        eGt2.append(eGraph2[0][t - WINDOW_SIZE :t])
        eGt2.append(eGraph2[1][t - WINDOW_SIZE :t])
        eGt2.append(eGraph2[2][t - WINDOW_SIZE :t, :])

        eGt3.append(eGraph3[0][t - WINDOW_SIZE :t])
        eGt3.append(eGraph3[1][t - WINDOW_SIZE :t])
        eGt3.append(eGraph3[2][t - WINDOW_SIZE :t, :])

        for i in range(len(size_Qc) - 1) :
            eQ1 = []
            eQ2 = []
            eQ3 = []
            eQ1.append(eQuery1[0][size_Qc[i] :size_Qc[i + 1]])
            eQ1.append(eQuery1[1][size_Qc[i] :size_Qc[i + 1]])
            eQ1.append(eQuery1[2][size_Qc[i] :size_Qc[i + 1],:])

            eQ2.append(eQuery2[0][size_Qc[i] :size_Qc[i + 1]])
            eQ2.append(eQuery2[1][size_Qc[i] :size_Qc[i + 1]])
            eQ2.append(eQuery2[2][size_Qc[i] :size_Qc[i + 1], :])

            eQ3.append(eQuery3[0][size_Qc[i] :size_Qc[i + 1]])
            eQ3.append(eQuery3[1][size_Qc[i] :size_Qc[i + 1]])
            eQ3.append(eQuery3[2][size_Qc[i] :size_Qc[i + 1], :])


            MatchsIDList1,MatchsIDList2,MatchsIDList3 = secTC_Process(eGt1,eGt2,eGt3, eQ1,eQ2,eQ3,NL)
            #MatchsTimeList.append(MatchsTime)
            #MatchsIDList.append(MatchsID)




def secTC_Process(Gt1,Gt2,Gt3, eQ1,eQ2,eQ3,NL):
    epsilon=10
    delta=0.01
    Delta=secGetSensitive(Gt1[2], Gt2[2], Gt3[2],NL)


#----------------------Add dummy labels----------------------#

    for i in range(NL):
        number_Dummy_Label=mpc.draw_Laplace_noise(Delta,epsilon,delta)+\
                           mpc.draw_Laplace_noise(Delta,epsilon,delta)+mpc.draw_Laplace_noise(Delta,epsilon,delta)

        if number_Dummy_Label<0:
            number_Dummy_Label=0
        sID1=np.random.randint(MAX,size=number_Dummy_Label,dtype=np.int32)
        sID2=np.random.randint(MAX,size=number_Dummy_Label,dtype=np.int32)
        sID3=-1*np.ones(number_Dummy_Label,dtype=np.int32)-sID1-sID2
        Gt1[0]=np.hstack((Gt1[0],sID1))
        Gt2[0]=np.hstack((Gt2[0],sID2))
        Gt3[0]=np.hstack((Gt3[0],sID3))

        eID1 = np.random.randint(MAX, size=number_Dummy_Label, dtype=np.int32)
        eID2 = np.random.randint(MAX, size=number_Dummy_Label, dtype=np.int32)
        eID3 = -1 * np.ones(number_Dummy_Label, dtype=np.int32) - eID1 - eID2
        Gt1[1] = np.hstack((Gt1[1], eID1))
        Gt2[1] = np.hstack((Gt2[1], eID2))
        Gt3[1] = np.hstack((Gt3[1], eID3))

        labels1=np.random.randint(2,size=(number_Dummy_Label,NL),dtype=bool)
        labels2=np.random.randint(2,size=(number_Dummy_Label,NL),dtype=bool)
        labels3=np.zeros((number_Dummy_Label,NL),dtype=bool)
        labels3[:,i]=np.ones(number_Dummy_Label,dtype=bool)

        labels3=labels3^labels1^labels2

        Gt1[2] = np.vstack((Gt1[2], labels1))
        Gt2[2] = np.vstack((Gt2[2], labels2))
        Gt3[2] = np.vstack((Gt3[2], labels3))
    dummy_Timestamp=np.random.randint(len(Gt1[0]),size=(len(Gt1[0])-len(Gt1[3])),dtype=np.int32)
    Gt1[3]=np.hstack((Gt1[3],dummy_Timestamp))         #Add dummy timestamps



    match_ID_list1 = []
    match_ID_list2 = []
    match_ID_list3 = []

    for i in range(len(Gt1[0])):
        for j in range(len(eQ1[0])):
            if (Gt1[2][i,:]^Gt2[2][i,:]^Gt3[2][i,:]^eQ1[2][j,:]^eQ2[2][j,:]^eQ3[2][j,:]).any():
                match_ID_list1.append(Gt1[0][i])
                match_ID_list2.append(Gt1[0][i])
                match_ID_list3.append(Gt1[0][i])



    return match_ID_list1,match_ID_list2,match_ID_list3


import numpy as np
import networkx as nx
import syft as sy
import time

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

    def reveal(self, value_ptrs):
        # retrieve the result from the parties and decrypt it
        result = value_ptrs.get().sum().item()
        return result

class SubgraphSearch:
    def __init__(self, pattern, n_parties):
        self.pattern = pattern
        self.rss = ReplicatedSecretSharing(n_parties)

    def generate_pattern(self):
        # generate the subgraph pattern to be searched for
        pattern_matrix = np.array(self.pattern)
        pattern_ptrs = self.rss.share_matrix(pattern_matrix)
        return pattern_ptrs

    def subgraph_search(self, graph_ptrs, pattern_ptrs):
        # perform the subgraph search
        matched = []
        n_nodes = graph_ptrs.shape[0]
        for i in range(n_nodes):
            for j in range(n_nodes):
                subgraph_ptrs = graph_ptrs[i:i+pattern_ptrs.shape[0], j:j+pattern_ptrs.shape[1], :]
                if subgraph_ptrs.shape == pattern_ptrs.shape:
                    if self.rss.isomorphic(subgraph_ptrs, pattern_ptrs):
                        matched.append((i, j))
        return matched

    def encrypt_streaming_graph(self, streaming_graph):
        # encrypt the streaming graph using the replicated secret sharing technique
        n_nodes = streaming_graph.shape[0]
        encrypted_graph = []
        for i in range(n_nodes):
            encrypted_row = []
            for j in range(n_nodes):
                matrix_ptrs = self.rss.share_matrix(streaming_graph[i][j])
                encrypted_row.append(matrix_ptrs)
            encrypted_graph.append(encrypted_row)
        return encrypted_graph

    def process_streaming_graph(self, encrypted_streaming_graph, pattern_ptrs, window_size, time_limit):
        # process the encrypted streaming graph and perform the time-constrained subgraph search
        matched = []
        start_time = time.time()
        for i in range(len(encrypted_streaming_graph)):
            for j in range(len(encrypted_streaming_graph)):
                subgraph_ptrs_list = []
                for k in range(window_size):
                    if i+k >= len(encrypted_streaming_graph) or j+k >= len(encrypted_streaming_graph):
                    break
                subgraph_ptrs_list.append(encrypted_streaming_graph[i+k][j+k])
                if len(subgraph_ptrs_list) == window_size:
                    subgraph_ptrs = self.rss.combine_matrices(subgraph_ptrs_list)
                if self.rss.isomorphic(subgraph_ptrs, pattern_ptrs):
                    matched.append((i, j))
                if time.time() - start_time >= time_limit:
                    return matched
        return matched



def secGetSensitive(label1,label2,label3,NL):
    MASK=np.random.randint(MAX,dtype=np.int32)
    k0,k1=FSS.DCFGen(0-MASK,1)
    MAXLabel0=0
    MAXLabel1=0

    MINLabel0=len(label1)
    MINLabel1=len(label1)

    labels=np.zeros((NL,NL),dtype=np.bool)
    labels[range(NL),range(NL)]=1

    numLabel=np.zeros(NL,dtype=np.int32)
    for i in range(NL):
        matBool1=np.sum(label1*labels[i,:],axis=1).astype(bool)
        matBool2=np.sum(label2*labels[i,:],axis=1).astype(bool)
        matBool3=np.sum(label3*labels[i,:],axis=1).astype(bool)

        matAirthe1,matAirthe2,matAirthe3=mpc.B2A(matBool1,matBool2,matBool3)

        sum1=np.sum(matAirthe1)
        sum2=np.sum(matAirthe2)
        sum3=np.sum(matAirthe3)

        r0=FSS.DCFEval(0,k0,MAXLabel0+MAXLabel1-sum1-sum2-sum3-MASK)
        r1=FSS.DCFEval(1,k1,MAXLabel0+MAXLabel1-sum1-sum2-sum3-MASK)

        tmp0,tmp1=mpc.ASS_MUL(MAXLabel0,r0,MAXLabel1,r1)
        MAXLabel0,MAXLabel1=mpc.ASS_MUL(sum1+sum2,1-r0,sum3,-r1)

        MAXLabel0=MAXLabel0+tmp0
        MAXLabel1=MAXLabel1+tmp1



        r0 = FSS.DCFEval(0, k0, sum1 + sum2 + sum3-MINLabel0 - MINLabel1 - MASK)
        r1 = FSS.DCFEval(1, k1, sum1 + sum2 + sum3-MINLabel0 - MINLabel1 - MASK)

        tmp0, tmp1 = mpc.ASS_MUL(sum1 + sum2, 1-r0, sum3, -r1)
        MINLabel0, MINLabel1 = mpc.ASS_MUL(MINLabel0, r0,MINLabel1 , r1)

        MINLabel0 = MINLabel0 + tmp0
        MINLabel1 = MINLabel1 + tmp1



    return MAXLabel0+MAXLabel1-MINLabel0-MINLabel1



def Pattern_Detection(G, WINDOW_SIZE, Q, Order, size_Qc) :
    for t in range(WINDOW_SIZE, len(G)) :
        time_start = time.time()
        MatchsTimeList=[]
        MatchsIDList=[]
        Gt = G[t - WINDOW_SIZE :t, :]  # A snapshot at time point t
        for i in range(len(size_Qc) - 1) :
            MatchsTime,MatchsID = TC_Process(Gt, Q[size_Qc[i] :size_Qc[i + 1], :])
            MatchsTimeList.append(MatchsTime)
            MatchsIDList.append(MatchsID)


        MatchsTime=MatchsTimeList[0]
        MatchsID=MatchsIDList[0]
        for i in range(1,len(MatchsTimeList)):
            if MatchsTime==[]:
                break
            MatchsTime,MatchsID=Compatibility_Test(MatchsTime,MatchsID,Q[:size_Qc[i],:],
                                                   MatchsTimeList[i],MatchsIDList[i],Q[size_Qc[i]:size_Qc[i+1],:],
                                                  Order)
            if MatchsTime==[]:
                break
        time_end = time.time()
        print('Snapshot:'+str(t))
        print('Time cost:', time_end - time_start, 's')
        print("The number of matches is "+str(len(MatchsTime)))

def TC_Process(Gt, Qc) :
    match_label_list= []
    match_ID_list = []
    for i in range(len(Qc)) :
        index=np.where(Gt[:, 2] == Qc[i, 2])[0] #find the indexes match the label Qc[i, 2])[0]; the index "[0]" is to achieve the resilt in numpy
        match_label_list.append(index)
        match_ID_list.append(Gt[index,0:2])

    MatchsTime,MatchsID=Time_Constraint_and_Structure_Test(match_label_list[:math.floor(len(match_label_list) / 2)],
                                match_ID_list[:math.floor(len(match_ID_list) / 2)], Qc[:math.floor(len(Qc) / 2), :2],
                                match_label_list[math.floor(len(match_label_list) / 2) :],
                                match_ID_list[math.floor(len(match_ID_list) / 2) :],Qc[math.floor(len(Qc) / 2):, :2])


    return MatchsTime,MatchsID

def Compatibility_Test(MatchsTime1,MatchsID1,Q1,MatchsTime2,MatchsID2,Q2,Order):
    if MatchsTime1==[] or MatchsID1==[] or MatchsTime2==[] or MatchsID2==[]:
        return [],[]



    MatchsTime = []
    MatchsID = []
    flag = 1
    for i in range(len(MatchsTime1)):
        for j in range(len(MatchsTime2)):
            Time1=MatchsTime1[i]
            Time2=MatchsTime2[j]
            ID1=MatchsID1[i]
            ID2=MatchsID2[j]
            if not testStructure(ID1, Q1, ID2, Q2) :
                continue
            isCompatible=1
            for m in range(len(Time1)):
                for n in range(len(Time2)):
                    if Time1[m]>Time2[n] and Order[m,n+len(Time1)]==1:
                        isCompatible = 0
                        break
                    if Time1[m]<Time2[n] and Order[m,n+len(Time1)]==2:
                        isCompatible = 0
                        break
                if not isCompatible:
                    break

            if isCompatible:
                tmpTime = np.zeros((len(Q1) + len(Q2), 1), dtype=int)
                tmpID = np.zeros((len(Q1) + len(Q2), 2), dtype=int)
                tmpTime[:len(Q1),:] = Time1
                tmpTime[len(Q1) :,:] = Time2
                MatchsTime.append(tmpTime)

                tmpID[:len(Q1),:] = ID1
                tmpID[len(Q1):,:] = ID2
                MatchsID.append(tmpID)

    return  MatchsTime,MatchsID




def Time_Constraint_and_Structure_Test(match_label_list1, match_ID_list1,Qc1,match_label_list2,match_ID_list2,Qc2) :
    if len(match_label_list1) == 1 :
        Ltime1 = match_label_list1[0].reshape((len(match_label_list1[0]),1)).tolist()
        LID1=match_ID_list1[0]

    else :
        Ltime1,LID1 = Time_Constraint_and_Structure_Test(match_label_list1[:math.floor(len(match_label_list1) / 2)],
                                match_ID_list1[:math.floor(len(match_ID_list1) / 2)],Qc1[:math.floor(len(Qc1) / 2), :2],
                                match_label_list1[math.floor(len(match_label_list1) / 2) :],
                                match_ID_list1[math.floor(len(match_ID_list1) / 2) :],Qc1[math.floor(len(Qc1) / 2):, :2])


    if len(match_label_list2) == 1 :
        Ltime2 = match_label_list2[0].reshape((len(match_label_list2[0]),1)).tolist()
        LID2=match_ID_list2[0]
    else :
        Ltime2,LID2 = Time_Constraint_and_Structure_Test(match_label_list2[:math.floor(len(match_label_list2) / 2)],
                                match_ID_list2[:math.floor(len(match_ID_list2) / 2)],Qc2[:math.floor(len(Qc2) / 2), :2],
                                match_label_list2[math.floor(len(match_label_list2) / 2) :],
                                match_ID_list2[math.floor(len(match_ID_list2) / 2) :],Qc2[math.floor(len(Qc2) / 2):, :2])
    if LID1==[] or LID2==[]:
        return [],[]


    MatchsTime=[]
    MatchsID=[]


    for i in range (len(Ltime1)):
        for j in range (len(Ltime2)):
            if Ltime1[i][-1]<Ltime2[j][0]: # if the timing orders satisfy the time constrains
                if  not testStructure(LID1[i],Qc1,LID2[j],Qc2):
                    continue

                tmpTime = np.zeros((len(Qc1) + len(Qc2), 1), dtype=int)
                tmpID = np.zeros((len(Qc1) + len(Qc2), 2), dtype=int)
                tmpTime[:len(Qc1),:]=Ltime1[i]
                tmpTime[len(Qc1):,:]=Ltime2[j]
                MatchsTime.append(tmpTime)


                tmpID[:len(Qc1),:]=LID1[i]
                tmpID[len(Qc1):,:]=LID2[j]
                MatchsID.append(tmpID)

    return  MatchsTime,MatchsID

def testStructure(LID1,Qc1,LID2,Qc2):

    if LID1.ndim==1:
        LID1=LID1.reshape(1,2)

    if LID2.ndim==1:
        LID2=LID2.reshape(1,2)

    mappings1={}
    mappings2={}


    for i in range(len(LID1)):
        if LID1[i,0] in mappings1:
            if mappings1[LID1[i,0]]!=Qc1[i,0]:
                return False
        else:
            mappings1[LID1[i,0]]=Qc1[i,0]

        if Qc1[i,0] in mappings2:
            if mappings2[Qc1[i,0]]!=LID1[i,0]:
                return False
        else:
            mappings2[Qc1[i,0]]=LID1[i,0]


        if LID1[i,1] in mappings1:
            if mappings1[LID1[i,1]]!=Qc1[i,1]:
                return False
        else:
            mappings1[LID1[i,1]]=Qc1[i,1]

        if Qc1[i, 1] in mappings2:
            if mappings2[Qc1[i, 1]] != LID1[i, 1]:
                return False
        else :
            mappings2[Qc1[i, 1]] = LID1[i, 1]

    AdjacencyMatrix(mappings1,mappings2)

    for i in range(len(LID2)):
        if LID2[i,0] in mappings1:
            if mappings1[LID2[i,0]]!=Qc2[i,0]:
                return False
        else:
            mappings1[LID2[i,0]]=Qc2[i,0]

        if Qc2[i,0] in mappings2:
            if mappings2[Qc2[i,0]]!=LID2[i,0]:
                return False
        else:
            mappings2[Qc2[i,0]]=LID2[i,0]

        if LID2[i,1] in mappings1:
            if mappings1[LID2[i,1]]!=Qc2[i,1]:
                return False
        else:
            mappings1[LID2[i,1]]=Qc2[i,1]

        if Qc2[i,1] in mappings2:
            if mappings2[Qc2[i,1]]!=LID2[i,1]:
                return False
        else:
            mappings2[Qc2[i,1]]=LID2[i,1]


    return True




# initialize hook for PySyft
hook = syft.TorchHook(torch)

# create workers for parties
bob = syft.VirtualWorker(hook, id="bob")
alice = syft.VirtualWorker(hook, id="alice")

# define matrix sizes
shape = (64, 32)

# create matrices using PySyft's SecureNN
A = (torch.rand(shape) * 100).fix_prec().share(bob, alice)
B = (torch.rand(shape) * 100).fix_prec().share(bob, alice)
C = torch.zeros(shape).fix_prec().share(bob, alice)

# move matrices to GPU
A = A.cuda()
B = B.cuda()
C = C.cuda()

# define the CUDA kernel
@torch.jit.script
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B"""
    C = C.private_tensor.child.child
    C.zero_()
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]

# perform matrix multiplication
matmul(A, B, C)

# move result back to CPU
C = C.get().float_prec()



hook = syft.TorchHook(torch)

# create workers for parties
bob = syft.VirtualWorker(hook, id="bob")
alice = syft.VirtualWorker(hook, id="alice")

# define the input data size and create the input tensor
input_size = 10
input_tensor = torch.randint(low=0, high=100, size=(input_size,))

# share the input tensor between the parties
shared_input_tensor = input_tensor.fix_prec().share(bob, alice)

# define the sorting function
def sort(shared_tensor):
    # get the unshared tensor for local computation
    tensor = shared_tensor.get().float_prec()
    # sort the tensor locally
    sorted_tensor = torch.sort(tensor)[0]
    # share the sorted tensor
    shared_sorted_tensor = sorted_tensor.fix_prec().share(bob, alice)
    return shared_sorted_tensor

# sort the input tensor
sorted_tensor = sort(shared_input_tensor)

# get the unshared tensor for printing
unshared_sorted_tensor = sorted_tensor.get().float_prec()

class AdjacencyMatrix :
    def __init__(self, adj_matrix_set, query_adj_matrix) :
        self.adj_matrix_set = adj_matrix_set
        self.query_adj_matrix = query_adj_matrix
        self.n_parties = 3
        self.hook = sy.TorchHook(torch)  # initialize PySyft hook
        self.crypto_provider = sy.VirtualWorker(self.hook,
                                                id='crypto_provider')  # create a virtual worker for crypto functions

    def combine_and_check(self) :
        # create pointers to adjacency matrices in the replicated secret sharing domain
        adj_ptrs = []
        for adj_matrix in self.adj_matrix_set :
            adj_ptrs.append(sy.MultiPointerTensor([torch.from_numpy(adj_matrix)], shape=(1, 3, 3)).share(bob, alice,
                                                                                                         self.crypto_provider))

        # combine adjacency matrices from different sets
        combined_adj_ptr = adj_ptrs[0].logical_or(*adj_ptrs[1 :])

        # get the combined adjacency matrix as a NumPy array
        combined_adj_matrix = combined_adj_ptr.get()[0].get()

        # check if the combined adjacency matrix matches the query adjacency matrix
        if np.array_equal(combined_adj_matrix, self.query_adj_matrix) :
            return True
        else :
            return False

