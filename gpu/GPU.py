from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states,xoroshiro128p_next
from numba import (config, cuda, float32, float64, uint32, int64, uint64,from_dtype, jit)

import numpy as np
N=64   #FSS输入值的长度
SCALE=1000000



@cuda.jit
def DPFEval_GPU(states,b,x_array,key,gpu_result):   #GPU上的FSS计算

    idx=cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if(idx<len(x_array)): #防止多余计算
        x = x_array[idx,:]
        s = key[0]
        t = b
        for i in range(N) :

            sCW = key[i * 3 + 1]
            tLCW = key[i * 3 + 2]
            tRCW = key[i * 3 + 3]

            if t :
                sL = (states[s]['s0']+states[s]['s1']) %SCALE ^ sCW
            else :
                sL = (states[s]['s0']+states[s]['s1'])% SCALE

            tL = (states[s+1]['s0']+states[s+1]['s1']) % 2 ^ (t & tLCW)

            if t :
                sR = (states[s+2]['s0']+states[s+2]['s1']) % SCALE ^ sCW
            else :
                sR = (states[s+2]['s0']+states[s+2]['s1']) % SCALE

            tR = ((states[s+3]['s0']+states[s+3]['s1'])  % 2) ^ (t & tRCW)

            if ~x[i] :
                s = sL
                t = tL
            else :
                s = sR
                t = tR
        c=(states[s]['s0']+states[s]['s1'])
        gpu_result[idx]=(-2 * b + 1) * (c+ t * key[-1])

def DPFGen(alpha,beta):

    alpha=num2Bool(alpha) #将alpha分解位bit-string

    states = create_xoroshiro128p_states(SCALE*10, seed=1)#这里确保Gen和Eval使用同一个随机数种子
    key0 = np.zeros(N * 3 + 2, np.dtype('int64'))
    key1 = np.zeros(N * 3 + 2, np.dtype('int64'))
    key0[0] = (int64(states[1]['s0']) + int64(states[1]['s1'])) % SCALE
    key1[0] = (int64(states[2]['s0']) + int64(states[2]['s1'])) % SCALE
    t0 = 0
    t1 = 1
    s0 = key0[0]
    s1 = key1[0]
    FLAG = 1
    for i in range(N) :
        sL0 = (int64(states[s0]['s0']) + int64(states[s0]['s1'])) % SCALE
        tL0 = (int64(states[s0 + 1]['s0']) + int64(states[s0 + 1]['s1'])) % 2
        sR0 = (int64(states[s0 + 2]['s0']) + int64(states[s0 + 2]['s1'])) % SCALE
        tR0 = (int64(states[s0 + 3]['s0']) + int64(states[s0 + 3]['s1'])) % 2

        sL1 = (int64(states[s1]['s0']) + int64(states[s1]['s1'])) % SCALE
        tL1 = (int64(states[s1 + 1]['s0']) + int64(states[s1 + 1]['s1'])) % 2
        sR1 = (int64(states[s1 + 2]['s0']) + int64(states[s1 + 2]['s1'])) % SCALE
        tR1 = (int64(states[s1 + 3]['s0']) + int64(states[s1 + 3]['s1']))% 2

        if alpha[i] :
            sLose0 = sL0
            sLose1 = sL1
            sKeep0 = sR0
            sKeep1 = sR1
        else :
            sLose0 = sR0
            sLose1 = sR1
            sKeep0 = sL0
            sKeep1 = sL1

        sCW = sLose0^sLose1
        tLCW = tL0 ^ tL1 ^ alpha[i] ^ 1
        tRCW = tR0 ^ tR1 ^ alpha[i]

        if alpha[i] :
            tKeep0 = tR0
            tKeep1 = tR1
            tKeepCW = tRCW
        else :
            tKeep0 = tL0
            tKeep1 = tL1
            tKeepCW = tLCW

        key0[FLAG] = sCW
        key1[FLAG] = sCW

        FLAG = FLAG + 1
        key0[FLAG] = tLCW
        key1[FLAG] = tLCW

        FLAG = FLAG + 1
        key0[FLAG] = tRCW
        key1[FLAG] = tRCW

        FLAG = FLAG + 1

        if t0 :
            s0 = sKeep0 ^ sCW
        else :
            s0 = sKeep0

        if t1 :
            s1 = sKeep1 ^ sCW

        else :
            s1 = sKeep1
        t0 = tKeep0 ^ (t0 & tKeepCW)
        t1 = tKeep1 ^ (t1 & tKeepCW)

    c1 = (int64(states[s0]['s0']) + int64(states[s0]['s1']))
    c2 = (int64(states[s1]['s0']) + int64(states[s1]['s1']))

    key0[FLAG] = (-2 * t1 + 1) * (beta - c1 + c2)
    key1[FLAG] = (-2 * t1 + 1) * (beta - c1 + c2)


    return key0,key1


def DPFEval(b,key,x_array):
    result=np.zeros(len(x_array), np.dtype('int64')) #GPU计算结果，与输入数组x_array的长度一样
    x_array=num2Bool_array(x_array)
    threads_per_block = 128
    blocks = int(np.ceil(len(x_array)/threads_per_block))
    rng_states = create_xoroshiro128p_states(SCALE*10, seed=1)#这里确保Gen和Eval使用同一个随机数种子
    x_array_GPU = cuda.to_device(x_array)
    key_GPU = cuda.to_device(key)
    gpu_result = cuda.to_device(result)#在GPU上初始化一块内存存放计算结果
    DPFEval_GPU[blocks,threads_per_block](rng_states,b,x_array_GPU,key_GPU,gpu_result)
    cuda.synchronize()
    result=gpu_result.copy_to_host()
    return result

def num2Bool(alpha):
    a = bin(alpha).replace('0b', '')[::-1]
    alpha = np.zeros(N, dtype=bool)
    for i in range(len(a)):
        if a[i]=='1':
            alpha[i]=True
    return alpha



def num2Bool_array(alpha_array):
    res=np.zeros((len(alpha_array),N),dtype=bool)
    for i in range(len(alpha_array)):
        a = bin(alpha_array[i]).replace('0b', '')[::-1]
        for j in range(len(a)):
            if a[j]=='1':
                res[i][j]=True

    return res