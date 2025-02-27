from datetime import datetime, timedelta
import numpy as np
import random
import sys
import pickle
import networkx as nx
import time 
import graph as utils
import operator
import argparse
import baseline
import fastmin_v2
import fastmink
from typing import List, Tuple, Dict
import ilp_version as ilp

      
if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("algorithm", help="baseline")
    parser.add_argument("algorithm", help="heuristic1")
    parser.add_argument("-k", help="number of intervals", default=10, type=int)
    parser.add_argument("--intlen", help="length of each active interval", default=10, type=int)
    parser.add_argument("--overlap", help="activity intervals overlap parameter, between 0 and 1", default=0.5, type=float)
    parser.add_argument("--nnodes", help="number of nodes in the graph", default=100, type=int)
    args = parser.parse_args()

    alg = args.algorithm
    k = args.k
    event_length = args.intlen
    overlap = args.overlap
    num_nodes = args.nnodes

    
    print('algorithm:', alg)
    print('number of intervals:', k)
    print('set event length:', event_length)
    print('set event overlap:', overlap)
    
    G = utils.generateGraph(n=num_nodes)
    print('number of nodes in the background network:', G.number_of_nodes())
    print('number of edges in the background network:', G.number_of_edges())
       
    timestamps, active_truth = utils.generateIntervals(G, event_length=event_length, overlap=overlap, number_intervals=k)
    
    print('number of timestamps', len(timestamps))

    if alg == 'baseline':
        Xstart, Xend = baseline.kbaseline(timestamps, k)
    elif alg == 'fastmin':
        Xstart, Xend = baseline.kbaseline(timestamps, k)
        Xstart2, Xend2 = fastmin_v2.proper_search(timestamps, k)
        Xstart3, Xend3 = fastmink.local_search(timestamps, k)
    else:
        print(f"Unknown algorithm: {alg}")
        sys.exit(1)

    if 'Xstart' in locals() and 'Xend' in locals():

        print('relative total length of solution =', utils.getCost(Xstart, Xend)/((event_length-1)*num_nodes))
        print('relative total length of solution with proper =', utils.getCost(Xstart2, Xend2)/((event_length-1)*num_nodes))
        print('relative total length of solution with local =', utils.getCost(Xstart3, Xend3)/((event_length-1)*num_nodes))
        print('total length by active truths', fastmink.calculate_length_for_active_truths(active_truth))
        print('total length ', utils.getCost(Xstart, Xend))
        print('total_length_of_proper_search = ',utils.getCost(Xstart2, Xend2))
        print('total_length_of_local_search = ',utils.getCost(Xstart3, Xend3))
        print('relative maximum length of solution =', utils.getMax(Xstart, Xend)/(event_length-1))
        print('relative maximum length of solution with fastmin =', utils.getMax(Xstart2, Xend2)/(event_length-1))
        p, r, f = utils.compareGT(Xstart, Xend, active_truth, timestamps)

        p1, r1, f1 = utils.compareGT(Xstart2, Xend2, active_truth, timestamps)

        print(f"precision = {p}, precision for proper search {p1}")
        
        print(f"recall = {r}, recall for proper search {r1}") 
        print(f"fmeasure = {f}, fmeaseure for proper search {f1}")
        for i in range(3):
            print('\n')
        m,x = ilp.ilp(timestamps, G, k)
        m.optimize()
        total_activations = m.objVal
        print(total_activations)
        active_intervals = ilp.active_intervals(m,x)
        p2, r2, f2 = utils.compare_intervals(active_intervals, active_truth, timestamps)

        print(f"precision = {p}, precision for ILP {p2}")
        
        print(f"recall = {r}, recall for ILP {r2}") 
        print(f"fmeasure = {f}, fmeaseure for ILP {f2}")