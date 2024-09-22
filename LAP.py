import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import networkx as nx

def draw_Laplace_noise(Delta,epsilon,delta):



    mu = -1 * Delta * (math.log((np.exp(epsilon / Delta) + 1) * (1 - np.power(1-delta, 1 / Delta)))) / epsilon
    #print(mu)
    mu = -1 * Delta * (math.log((np.exp(epsilon / Delta) + 1) * (1 - np.sqrt(1 - delta)))) / epsilon

    print(mu)


draw_Laplace_noise(500,1.6,0.00001)
draw_Laplace_noise(500,1.4,0.00001)
draw_Laplace_noise(500,1.2,0.00001)
draw_Laplace_noise(500,1,0.00001)
draw_Laplace_noise(500,0.8,0.00001)

