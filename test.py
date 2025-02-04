from datetime import datetime, timedelta
import numpy as np
import copy
import sys
import pickle
import graph as utils
import operator
from typing import List, Tuple, Dict
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("algorithm", help="baseline")
parser.add_argument("-k", help="number of intervals", default=10, type=int)
parser.add_argument("--intlen", help="length of each active interval", default=10, type=int)
parser.add_argument("--overlap", help="activity intervals overlap parameter, between 0 and 1", default=0.5, type=float)
parser.add_argument("--nnodes", help="number of nodes in the graph", default=10, type=int)
args = parser.parse_args()

alg = args.algorithm
k = args.k
event_length = args.intlen
overlap = args.overlap
num_nodes = args.nnodes

G = utils.generateGraph(n=num_nodes)
timestamps, active_truth = utils.generateIntervals(G, event_length=event_length, overlap=overlap, number_intervals=k)
    
print('number of timestamps', len(timestamps))

#print(utils.neighborMapping(timestamps))