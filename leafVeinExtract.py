import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import cv2


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Edge(object):
    def __init__(self, start_node, end_node, slope, width, length,startNodeID,endNodeID):
        self.startNode = start_node
        self.endNode = end_node
        self.slop = slope
        self.width = width
        self.length = length
        self.startNodeID = startNodeID
        self.endNodeID = endNodeID

def plot_graph(graph, slope_threshold):
    dpi = 600
    figsize = (12, 12)
    f = plt.figure(dpi=dpi, figsize=figsize)
    plt.clf()
    img = plt.imread('LDC001_2_1_t_b.png')
    plt.imshow(img)
    plt.gca().invert_yaxis()
    node_collection = dict(graph.nodes(data=True))
    edge_collection = graph.edges(data=True)
    filtered_edge_list = []
    sorted_edge = sorted(edge_collection, key=lambda edge:edge[2]['conductivity'], reverse=True)
    widest_edge = sorted_edge[0]
    start_x = node_collection[widest_edge[0]]['x']
    start_y = 4800-node_collection[widest_edge[0]]['y']
    end_x = node_collection[widest_edge[1]]['x']
    end_y = 4800-node_collection[widest_edge[1]]['y']
    # get filtered edge list
    for edge in sorted_edge:
        x1 = node_collection[edge[0]]['x']
        y1 = 4800-node_collection[edge[0]]['y']
        x2 = node_collection[edge[1]]['x']
        y2 = 4800-node_collection[edge[1]]['y']
        k = abs((y2 - y1)/((x2 - x1) + 0.001))
        if 1 < edge[2]['weight'] < 200 and x1 != x2 and k < 0.1 and abs((y1+y2)/2 - (start_y + end_y)/2) < 200:
            my_edge = Edge(Point(x1, y1), Point(x2, y2), k, edge[2]['conductivity'], edge[2]['weight'],edge[0],edge[1])
            filtered_edge_list.append(my_edge)
            #f.gca().plot([x1, x2], [y1, y2], linewidth=edge[2]['conductivity'], color='r', zorder=2)[0]


    # plot the widest edge which is used as the start edge
    f.gca().plot([start_x, end_x], [start_y, end_y], linewidth=12, color='k', zorder=3)[0]


    plt.show()
    print("max_width_edge", sorted_edge[0])


def extension(first_list,edge_list,node_collection,adj):
    # set terminate condition
    if first_list.size() > 4000:
        return first_list
    else:
        for edge in edge_list:
            x1 = node_collection[edge[0]]['x']
            y1 = 4800 - node_collection[edge[0]]['y']
            x2 = node_collection[edge[1]]['x']
            y2 = 4800 - node_collection[edge[1]]['y']
            k = abs((y2 - y1) / ((x2 - x1) + 0.001))
            length = edge[2]['weight']

            if edge[0] or edge[1]:
                pass

        pass




if __name__ == "__main__":
    graph = nx.read_gpickle('LDC001_2_1_t_b_graph_p5_r0.gpickle')
    plot_graph(graph, 1)

