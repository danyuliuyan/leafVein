#!/usr/bin/python
# -*- coding:utf8 -*-
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gc
from scipy import stats
import psutil
import math
DISTANCE_THRESHOLD = 5
#需要在使用这个变量之前获取最大的长度，从而去除错误的边
LENGTH_THRESHOLD = 1000
class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

def extension(edge_start,edge_collection,node_collection):
    for index, edge1 in enumerate(edge_collection):
        x1 = float(node_collection[edge1[0]]['x'])
        x2 = float(node_collection[edge1[1]]['x'])
        y1 = float(node_collection[edge1[0]]['y'])
        y2 = float(node_collection[edge1[1]]['y'])

        x1_in = float(node_collection[edge_start[0]]['x'])
        x2_in = float(node_collection[edge_start[1]]['x'])
        y1_in = float(node_collection[edge_start[0]]['y'])
        y2_in = float(node_collection[edge_start[1]]['y'])

        if edge_start[0] != edge1[0] and edge_start[1]!= edge1[1] and ((abs(x1-x1_in) < 5 and abs(y1-y1_in) < 5) or (abs(x2-x2_in)<5 and abs(y2-y2_in) < 5)):
            return edge1

def plot_graph(graph,threshold):

    memory_convert = 1024*1024
    mem = psutil.virtual_memory()
    print "mem start:\t"+ str((mem.total - mem.used)/memory_convert)+" M"
    dpi= 600
    figsize = (12,12)
    f = plt.figure(dpi=dpi, figsize=figsize)
    plt.clf()
    node_collection = dict(graph.nodes(data=True))
    edge_collection = graph.edges(data=True)
    # obtain max width and weight
    widths = np.array([edge[2]['conductivity'] for edge in edge_collection])
    weights = np.array([edge[2]['weight']for edge in edge_collection])
    max_weight = max(weights)
    max_width = max(widths)
    print "max length:\t" + str(max_weight)
    print "max width:\t" + str(max_width)
    point_list = []
    width_scale_factor = 1. / np.amax(widths) * 5
    width_list = []

    # obtain the mode of node y coordinate
    y_1 = np.array([node_collection[edge[0]]['y'] for edge in edge_collection])
    mode_y_coordinate =stats.mode(y_1)[0][0]
    #max_count_num = np.median(y_1)
    print "mode y:\t" + str(mode_y_coordinate)


    # obtain the mode of edge length 
    weight_1 = np.array([edge[2]['weight'] for edge in edge_collection])
    mode_weight = stats.mode(weight_1)[0][0]
    print "mode weight:\t" + str(mode_weight)
    # filteredEdges
    filteredEdge = [];
    # obtain the mode of edge width 
    width_1 = np.asarray([edge[2]['conductivity'] for edge in edge_collection])
    mode_width = stats.mode(width_1)[0][0]
    print "mode width:\t" + str(mode_width)
    plt.title("mode y: " + str(mode_y_coordinate)+" "+"mode weight: "+str(mode_weight)+" "+"mode width: " + str(mode_width))
    print "mem:\t"+ str((mem.total - mem.used)/memory_convert)+" M"

    for edge in edge_collection:
        if (mem.total - mem.used)/memory_convert < 200:
            exit(0)
        width = edge[2]['conductivity']
        weight = edge[2]['weight']
        x1 = float(node_collection[edge[0]]['x'])
        x2 = float(node_collection[edge[1]]['x'])
        y1 = float(node_collection[edge[0]]['y'])
        y2 = float(node_collection[edge[1]]['y'])
        k = abs((y2-y1)/(x2-x1+0.005))
        if (weight < 0.1 * max_weight and width > threshold * max_width and 100 < k):
            filteredEdge.append(edge)
            f.gca().plot([x1, x2], [y1, y2], linewidth=width_scale_factor * width, color='r', zorder=3)[0]


    sorted_edge = sorted(filteredEdge, key=lambda edge:edge[2]['conductivity'], reverse=True)
    widest_edge = sorted_edge[0]
    f.gca().plot([node_collection[widest_edge[0]]['x'],node_collection[widest_edge[1]]['x']],[node_collection[widest_edge[0]]['y'],node_collection[widest_edge[1]]['y']],zorder=4)[0]




    # find the main vein
    '''
    count = 0
    num = 0
    main_vein = []
    main_vein.append(widest_edge)
    while count <= 4:
        start = main_vein[-1]
        main_vein.append(extension(start,filteredEdge,node_collection))

    for edge in main_vein:
        x1 = float(node_collection[edge[0]]['x'])
        x2 = float(node_collection[edge[1]]['x'])
        y1 = float(node_collection[edge[0]]['y'])
        y2 = float(node_collection[edge[1]]['y'])
        f.gca().plot([x1, x2], [y1, y2], linewidth=5, color='g', zorder=3)[0]
    '''
    plt.show()


    f.canvas.draw_idle()
    plt.savefig("main_vein", format='png', dpi=600, pad_inches = 0)



#初步过滤起点和终点相同的边、孤立点
def filterEdge(edgeList):
    filteredEdge = []
    for edge in edgeList:
        startNode, endNode = edge[0], edge[1]
        if distancee(startNode, endNode) > DISTANCE_THRESHOLD and edge[2]['conductivity'] < 0.1 * LENGTH_THRESHOLD:
            filteredEdge.append(edge)


#计算任意两个节点之间的欧氏距离
def distancee(point1,point2,node_collection):
    x1 = float(node_collection[point1]['x'])
    x2 = float(node_collection[point2]['x'])
    y1 = float(node_collection[point1]['y'])
    y2 = float(node_collection[point2]['y'])
    return math.sqrt(math.pow(x1-x2, 2)+math.pow(y1-y2, 2))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="drawGraph -s gpickle_file_path -d width_threshold")
    #parser.add_argument("-s", help="the path of .gpickle file")
    # parser.add_argument("-t", help="threshold of edge width")
    # args = parser.parse_args()
    # G = nx.read_gpickle(args.s)
    #plot_graph(G, float(args.t))
    G = nx.read_gpickle('LDC001_2_1_t_b_graph_p5_r0.gpickle')
    plot_graph(G,0.1)
