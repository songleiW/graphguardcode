import sys
import time
import numpy as np
import Query_Process
import fileIO
import mpc
import math
import Encryption
import secQuery_Process
import numpy as np
from numba import cuda


flag = 2
query_size = 12
WINDOW_SIZE = 50000
if flag:
    NL=97
else:
    NL=2
G = fileIO.readGraph(flag)  # read the dynamic graph

eGraph1,eGraph2,eGraph3=Encryption.encGraph(G[:WINDOW_SIZE*10,:],NL)

Query, Order, size_Qc = fileIO.readQuery(flag, query_size)  # read the query
eQuery1,eQuery2,eQuery3=Encryption.encQuery(Query, Order,NL)

secQuery_Process.secPattern_Detection(eGraph1,eGraph2,eGraph3, WINDOW_SIZE, eQuery1,eQuery2,eQuery3, size_Qc,NL)  # process a query


#Query_Process.Pattern_Detection(G, WINDOW_SIZE, Q, Order, size_Qc)  # process a query
