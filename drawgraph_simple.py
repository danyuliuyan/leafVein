import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

def plot_graph(graph,threshold):
    dpi= 100
    figsize = (18,12)
    f = plt.figure(dpi=dpi, figsize=figsize)
   
    plt.clf()
    node_collection = dict(graph.nodes(data=True))
    edge_collection = graph.edges(data=True)
    widths = np.array([edge[2]['conductivity'] for edge in edge_collection])
    weights = np.array([edge[2]['weight']for edge in edge_collection])
    max_weight = max(weights)
    max_width = max(widths)
    width_scale_factor = 1. / np.amax(widths) * 5
    print "max length:\t" + str(max_weight)
    print "max width:\t" + str(max_width)
    for edge in edge_collection:
        width = edge[2]['conductivity']
        weight = edge[2]['weight']
        x1 = float(node_collection[edge[0]]['x'])
        x2 = float(node_collection[edge[1]]['x'])
        y1 = float(node_collection[edge[0]]['y'])
        y2 = float(node_collection[edge[1]]['y'])

        if width == max_width:
            f.gca().plot([x1, x2], [y1, y2], linewidth=width_scale_factor * width, color='b', zorder=3)[0]



    '''
    for node in node_collection:
        x = node_collection[node]['x']
        y = node_collection[node]['y']
        new_node = f.gca().plot(x, y, marker='o', color='b', markersize=2, zorder=9)[0]
        node_list.update({node: new_node})
    '''
    f.canvas.draw_idle()
    plt.savefig("main"+str(threshold),format="png")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="drawGraph -s gpickle_file_path -d width_threshold")
    parser.add_argument("-s", help="the path of .gpickle file")
    parser.add_argument("-t", help="threshold of edge width")
    args = parser.parse_args()
    G = nx.read_gpickle(args.s)
    plot_graph(G, float(args.t))
