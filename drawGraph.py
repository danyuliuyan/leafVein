import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gc
from scipy import stats
import psutil
import os
import sys
class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

def plot_graph(graph,threshold):

    memory_convert = 1024*1024;
    mem = psutil.virtual_memory()
    print "mem start:\t"+ str((mem.total - mem.used)/memory_convert)+" M"
    dpi= 100
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
        if weight < 0.1 * max_weight: 
            f.gca().plot([x1, x2], [y1, y2], linewidth=width_scale_factor * width, color='r', zorder=3)[0]
        sorted_edge = sorted(edge_collection, key=lambda edge:edge[2]['conductivity'], reverse=True)
        widest_edge = sorted_edge[0]
        
        f.gca().plot([node_collection[widest_edge[0]]['x'],node_collection[widest_edge[1]]['x']],[node_collection[widest_edge[0]]['y'],node_collection[widest_edge[1]]['y']],zorder=4)[0]
    #pl = sorted(point_list,key= lambda point : point.x)

    #z1 = np.polyfit([point.x for point in pl],[point.y for point in pl],2)
    #p1 = np.poly1d(z1)
    #print p1
    #y_vals = p1([point.x for point in pl])
    #plt.plot([point.x for point in pl],y_vals,'b')
    del x1, x2,width_list
    
    
    for edge in edge_collection:
        if (mem.total - mem.used)/memory_convert < 200:
            exit(0)
        x1 = node_collection[edge[0]]['x']
        x2 = node_collection[edge[1]]['x']
        y1 = node_collection[edge[0]]['y']
        y2 = node_collection[edge[1]]['y']
        width = edge[2]['conductivity']
        width_list.append(width)
        if width > threshold * max_width:
            print "mem:\t" + str((mem.total - mem.used)/memory_convert)+" M"
            f.gca().plot([x1, x2], [y1, y2], linewidth=width_scale_factor * width, color='r', zorder=2,)[0]
            #edge_list.update({(edge[0], edge[1]): new_edge})
    del node_collection,edge_collection
    
    
    '''
    for node in node_collection:
        x = node_collection[node]['x']
        y = node_collection[node]['y']
        new_node = f.gca().plot(x, y, marker='o', color='b', markersize=2, zorder=9)[0]
        node_list.update({node: new_node})
    '''
    f.canvas.draw_idle()
    plt.savefig("main_vein", format='png', dpi=600, pad_inches = 0)
    #plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="drawGraph -s gpickle_file_path -d width_threshold")
    parser.add_argument("-s", help="the path of .gpickle file")
    parser.add_argument("-t", help="threshold of edge width")
    args = parser.parse_args()
    G = nx.read_gpickle(args.s)
    plot_graph(G, float(args.t))

