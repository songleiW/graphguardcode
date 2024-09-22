import math
import time
import numpy as np
import syft as sy
from numba import cuda


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

