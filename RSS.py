import numpy as np
import random
import string
import sys
LAMBDA = 128
import math
N=32
MAX = 2147483647#sys.maxsize/10
import syft as sy

def DPFGen(alpha,beta):
    alpha=num2Bool(alpha)

    key0= np.zeros(N*3+2, np.dtype('int64'))
    key1= np.zeros(N*3+2, np.dtype('int64'))


    random.seed(23452345)
    key0[0]=random.randint(0, MAX)
    random.seed(56785678)
    key1[0] = random.randint(0, MAX)

    t0=0
    t1=1

    s0=key0[0]
    s1=key1[0]

    FLAG=1

    for i in range(N):

        random.seed(s0)
        sL0=random.randint(0, MAX)
        tL0=random.randint(0, MAX)%2
        sR0 = random.randint(0, MAX)
        tR0 = random.randint(0, MAX)%2

        random.seed(s1)
        sL1 = random.randint(0, MAX)
        tL1 = random.randint(0, MAX) % 2
        sR1 = random.randint(0, MAX)
        tR1 = random.randint(0, MAX) % 2

        if alpha[i]:
            sLose0=sL0
            sLose1=sL1
            sKeep0=sR0
            sKeep1=sR1
        else:
            sLose0 = sR0
            sLose1 = sR1
            sKeep0 = sL0
            sKeep1 = sL1

        sCW=sLose0^sLose1

        tLCW=tL0^tL1^alpha[i]^1
        tRCW=tR0^tR1^alpha[i]

        if alpha[i] :
            tKeep0=tR0
            tKeep1=tR1
            tKeepCW=tRCW
        else:
            tKeep0 = tL0
            tKeep1 = tL1
            tKeepCW = tLCW

        key0[FLAG]=sCW
        key1[FLAG]=sCW

        FLAG=FLAG+1
        key0[FLAG] = tLCW
        key1[FLAG] = tLCW

        FLAG = FLAG + 1
        key0[FLAG] = tRCW
        key1[FLAG] = tRCW


        FLAG = FLAG + 1



        if t0:
            s0=sKeep0^sCW
        else:
            s0=sKeep0

        if t1:
            s1=sKeep1^sCW

        else:
            s1=sKeep1
        t0=tKeep0^(t0&tKeepCW)
        t1=tKeep1^(t1&tKeepCW)

    random.seed(s0)
    c1=random.randint(0,MAX)

    random.seed(s1)
    c2 = random.randint(0, MAX)


    key0[FLAG]=(-2*t1+1)*(beta-c1+c2)
    key1[FLAG]=(-2*t1+1)*(beta-c1+c2)

    return key0,key1

def DPFEval(b,key,x):
    x=num2Bool(x)
    s=key[0]
    t=b
    for i in range(N):

        sCW=key[i*3+1]
        tLCW=key[i*3+2]
        tRCW=key[i*3+3]

        random.seed(s)

        if t:
            sL = random.randint(0, MAX)^ sCW
        else:
            sL = random.randint(0, MAX)

        tL = (random.randint(0, MAX) % 2)^(t&tLCW)

        if t:
            sR = random.randint(0, MAX)^ sCW
        else:
            sR = random.randint(0, MAX)

        tR = (random.randint(0, MAX) % 2)^(t&tRCW)

        if ~x[i]:
            s=sL
            t=tL
        else:
            s = sR
            t = tR

    random.seed(s)
    return (-2*b+1)*(random.randint(0, MAX)+t*key[-1])

def DCFGen(alpha,beta):

    alpha = num2Bool(alpha)[::-1]

    key0 = []
    key1 = []

    s0 = random01(LAMBDA, '00')
    s1 = random01(LAMBDA, '01')

    key0.append(s0)
    key1.append(s1)

    Valpha = 0


    t0 = False
    t1 = True

    for i in range(N) :
        skey=[]

        G = random01(4 * LAMBDA + 2, s0)

        sL0 = G[:LAMBDA]
        vL0 = G[LAMBDA :2 * LAMBDA]
        tL0 = str2Bool(G[2 * LAMBDA])
        sR0 = G[2 * LAMBDA + 1:3 * LAMBDA + 1]
        vR0 = G[3 * LAMBDA + 1 :4 * LAMBDA + 1]
        tR0 = str2Bool(G[4 * LAMBDA + 1])

        G = random01(4 * LAMBDA + 2, s1)

        sL1 = G[:LAMBDA]
        vL1 = G[LAMBDA :2 * LAMBDA]
        tL1 = str2Bool(G[2 * LAMBDA])
        sR1 = G[2 * LAMBDA + 1:3 * LAMBDA + 1]
        vR1 = G[3 * LAMBDA + 1 :4 * LAMBDA + 1]
        tR1 = str2Bool(G[4 * LAMBDA + 1])


        if alpha[i] :
            sLose0 = sL0
            sLose1 = sL1
            sKeep0 = sR0
            sKeep1 = sR1
            vLose0=vL0
            vLose1=vL1
            vKeep0=vR0
            vKeep1=vR1
            tKeep0=tR0
            tKeep1=tR1

        else :
            sLose0 = sR0
            sLose1 = sR1
            sKeep0 = sL0
            sKeep1 = sL1
            vLose0=vR0
            vLose1=vR1
            vKeep0=vL0
            vKeep1=vL1
            tKeep0 = tL0
            tKeep1 = tL1

        sCW=Bool2Str(str2Bool(sLose0) ^ str2Bool(sLose1))

        random.seed(vLose0)
        c1 = random.randint(0, 134545)

        random.seed(vLose1)
        c2 = random.randint(0, 134545)

        vCW=(-2*t1+1)*(c2-c1-Valpha)

        if alpha[i] :
            vCW=vCW+(-2*t1+1)*beta

        random.seed(vKeep1)

        Valpha=Valpha-random.randint(0, 134545)

        random.seed(vKeep0)
        Valpha=Valpha+random.randint(0, 134545)+(-2*t1+1)*vCW

        tLCW=tL0^tL1^alpha[i]^True
        tRCW=tR0^tR1^alpha[i]

        skey.append(sCW)
        skey.append(vCW)
        skey.append(tLCW)
        skey.append(tRCW)

        key0.append(skey)
        key1.append(skey)


        s0=Bool2Str(str2Bool(sKeep0)^t0&str2Bool(sCW))
        s1=Bool2Str(str2Bool(sKeep1)^t1&str2Bool(sCW))

        if alpha[i] :
            t0=tKeep0^t0&tRCW
            t1=tKeep1^t1&tRCW

        else:
            t0=tKeep0^t0&tLCW
            t1=tKeep1^t1&tLCW

    random.seed(s0)
    c1 = random.randint(0, 134545)

    random.seed(s1)
    c2 = random.randint(0, 134545)


    CW=(-2*t1+1)*(c2-c1-Valpha)

    key0.append(CW)
    key1.append(CW)

    return key0,key1

def DCFEval(b,key,x):
    s=key[0]
    V=0
    t=True if b==1 else False
    x=num2Bool(x)[::-1]
    for i in range(N):
        CW=key[i+1]
        sCW=CW[0]
        vCW=CW[1]
        tLCW=CW[2]
        tRCW=CW[3]

        G = random01(4 * LAMBDA + 2, s)

        s_L = G[:LAMBDA]
        v_L = G[LAMBDA :2 * LAMBDA]
        t_L = str2Bool(G[2 * LAMBDA])
        s_R = G[2 * LAMBDA + 1:3 * LAMBDA + 1]
        v_R = G[3 * LAMBDA + 1 :4 * LAMBDA + 1]
        t_R = str2Bool(G[4 * LAMBDA + 1])

        str1=s_L+Bool2Str(t_L)+s_R+Bool2Str(t_R)
        str2=sCW+Bool2Str(tLCW)+sCW+Bool2Str(tRCW)

        T=Bool2Str(str2Bool(str1)^(t&str2Bool(str2)))

        sL=T[:LAMBDA]
        tL=str2Bool(T[LAMBDA])
        sR=T[LAMBDA+1:2*LAMBDA+1]
        tR=str2Bool(T[2*LAMBDA+1])

        if not x[i]:
            random.seed(v_L)
            V=V+(-2*b+1)*(random.randint(0, 134545)+t*vCW)
            s=sL
            t=tL
        else:
            random.seed(v_R)
            V = V + (-2 * b + 1) * (random.randint(0, 134545) + t * vCW)
            s = sR
            t = tR
    random.seed(s)
    return V[0]+(-2 * b + 1)*(random.randint(0, 134545)+t*key[-1])[0]

def random01(length,seed):
    res=''
    for i in range(int(length/32)):
        random.seed(seed)
        res=res+bin(random.randint(44444444410, 54444444410))[2:2+32]
    if not len(res)==length:
        res = res + bin(random.randint(0, 44444444410))[2:2+length-len(res)]
    return res

def str2Bool(str):
    res=np.zeros(len(str), dtype=bool)
    for i in range(len(str)) :
        if str[i]=='1':
            res[i]=True
    return res

def Bool2Str(Boolean):
    res=''
    for i in range(len(Boolean)):
        if Boolean[i]:
            res=res+'1'
        else:
            res=res+'0'
    return res
def num2Bool(alpha):
    a = bin(alpha).replace('0b', '')[::-1]
    alpha = np.zeros(N, dtype=bool)
    for i in range(len(a)):
        if a[i]=='1':
            alpha[i]=True
    return alpha
