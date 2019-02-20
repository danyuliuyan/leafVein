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

MAX_LENGTH_THRESHOLD = 100
#需要在使用这个变量之前获取最大的长度，从而去除错误的边
MIN_LENGTH_THRESHOLD = 5
MAX_SLOPE_THRESHOLD = 0.1
#初始化画布供全局使用


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

#初步过滤起点和终点相同的边、孤立点 同时过滤掉斜率较大的边
def filterEdge(edgeList):
    filteredEdge = []
    for edge in edgeList:
        startNode, endNode = edge[0], edge[1]
        distance, slope = distance_slope(startNode, endNode)
        edge[3]['slope'] = slope
        if MIN_LENGTH_THRESHOLD < distance < 0.1 * MAX_LENGTH_THRESHOLD and slope < MAX_SLOPE_THRESHOLD:
            filteredEdge.append(edge)
    return filteredEdge


#计算任意两个节点之间的欧氏距离
def distance_slope(point1, point2):
    x1 = float(point1['x'])
    x2 = float(point2['x'])
    y1 = float(point1['y'])
    y2 = float(point2['y'])
    return math.sqrt(math.pow(x1-x2, 2)+math.pow(y1-y2, 2)),abs((y2-y1)/(x2-x1+0.0001))

#临时在原图中画出某一条边
def draw_edge(edge):
    x1, y1 = edge[0]['x'], edge[0]['y']
    x2, y2 = edge[1]['x'], edge[1]['y']
    plt.clf()
    axes = f.gca()
    axes.plot([x1, x2], [y1, y2], zorder=4, color='g',linewidth=edge[3]['conductivity'])[0]
    axes.set_xlim([1, 6000])
    axes.set_ylim([1, 4800])
    plt.show()


#处理原来的边将节点信息转换为坐标
def convert_nodeid_to_coordinate(edge_list, node_collection):
    converted_edgelist = []
    for edge in edge_list:
        temp_edge = [{'x': node_collection[edge[0]]['x'], 'y': node_collection[edge[0]]['y']}, {'x': node_collection[edge[1]]['x'],'y': node_collection[edge[1]]['y']},{'isUsed':False}, edge[2],edge[0],edge[1]]
        converted_edgelist.append(temp_edge)
    return converted_edgelist


#根据每一小段的公式 根据其斜率像两边延伸自身长度的一半，并记录下新的起点的终点的坐标
#根据起点和终点以及斜率确定直线
def extension(sorted_edge_list, start_edge):
    result = []
    start_edge_bak = start_edge
    for edge in sorted_edge_list:
        new_slope = abs((edge[0]['y'] - start_edge[0]['y'])/(edge[0]['x'] - start_edge[1]['x'] + 0.001))
        if abs(edge[0]['y'] - start_edge[1]['y']) > 20 or abs(edge[0]['y'] - start_edge_bak[0]['y']) > 20 or abs(new_slope - slope(start_edge)) > 2 or edge == start_edge_bak:
            continue
        slope1 = slope(edge)
        bias = edge[0]['y'] - slope1 *edge[0]['x']
        new_start_x, new_end_x = edge[0]['x']-2*math.floor(edge[3]['weight']/2), edge[1]['x'] + 2*math.floor(edge[3]['weight']/2)
        new_start_y, new_end_y = slope1 * new_start_x + bias, slope1 * new_end_x + bias
        edge[0] = {'x': new_start_x, 'y': new_start_y}
        edge[1] = {'x': new_end_x, 'y': new_end_y}
        result.append(edge)
        start_edge = edge
    points = []
    for edge in result:
        points.extend([Point(edge[0]['x'], 4800 - edge[0]['y'])])
    pointList = sorted(points, key=lambda point: point.x)
    x = [point.x for point in pointList]
    y = [point.y for point in pointList]
    dpi = 600
    figsize = (12, 12)
    f = plt.figure(dpi=dpi, figsize=figsize)
    plt.clf()
    axes = f.gca()
    axes.set_ylim([1, 4800])
    axes.set_xlim([1, 6000])
    plt.imshow(plt.imread('/home/zyp/network_extract_test/LDC001_images/LDC001_2_2_t_b/LDC001_2_2_t_b.png'))
    f.gca().plot(x, y, linewidth=2, color='g', zorder=3)[0]
    plt.show()
    return result


#修改后的主叶脉延伸
#
def extension2(sorted_edgelist,start_edge):
    count = 0
    num = 0
    result = []
    dpi = 600
    figsize = (12, 12)
    f = plt.figure(dpi=dpi, figsize=figsize)
    plt.clf()
    axes = f.gca()
    axes.set_ylim([1, 4800])
    axes.set_xlim([1, 6000])
    while(count < 100):
        for index,edge in enumerate(sorted_edge_list):
            new_slope = abs((edge[0]['y'] - start_edge[0]['y']) / (edge[0]['x'] - start_edge[1]['x'] +0.001))
            if abs(edge[0]['y'] - start_edge[1]['y']) < 50 and abs(edge[0]['x'] - start_edge[1]['x']) < 400 and new_slope < 4 and edge[2]['isUsed'] == False:
                result.append(edge)
                start_edge = edge
                edge[2]['isUsed'] = True
                f.gca().plot([edge[0]['x'],edge[1]['x']], [4800 - edge[0]['y'],  4800 - edge[1]['y']],linewidth=edge[3]['conductivity'], color='g', zorder=4)[0]
                break
            num = index
        if num >= len(sorted_edge_list) -1:
            break
        count += 1

    plt.imshow(plt.imread('/home/zyp/network_extract_test/LDC001_images/LDC001_1_2_t_b/LDC001_1_2_t_b.png'))
    plt.show()

    return result

#利用原来的邻接矩阵的关系来连接
def extension3(sorted_edge_by_x_coordinate):
    start_edge = sorted_edge_by_x_coordinate[0]
    step = 0
    result = []
    while(step < 100):
        for edge in sorted_edge_by_x_coordinate:
            if (start_edge[4] == edge[4] or start_edge[5] == edge[5] or distance(start_edge,edge)<50) and edge[2]['isUsed'] == False and -1<slope(edge)<1 and abs(edge[1]['y']-start_edge[1]['y']) < 50:
               edge[2]['isUsed'] = True
               result.append(edge)
               start_edge = edge
        step +=1
    return result
def slope(edge):
    return (edge[1]['y'] - edge[0]['y'])/(edge[1]['x']-edge[0]['x']+0.01)
#从右边向左边
def extension4(sorted_edge_by_x_coordinate):
    start_edge = sorted_edge_by_x_coordinate[0]
    #所有的线段都是右边的端点4横坐标更大

    temp_result = []
    count = 0
    m = 0
    for edge in sorted_edge_by_x_coordinate:
        count += 1
        start_node, end_node = start_edge[5], start_edge[4]
        if (edge[5] == end_node and abs(slope(edge)-slope(start_edge)) < 1 and edge[1]['x'] <= edge[0]['x'] and abs(edge[0]['y'] - start_edge[0]['y'] < 5)) or ((abs(edge[1]['x']-start_edge[0]['x'])<10 and edge[1]['x'] <= start_edge[0]['x'] and abs(edge[1]['y']-start_edge[0]['y']<5) and abs(edge[0]['y'] - start_edge[0]['y'] < 5)and edge[1]['x']>edge[0]['x'] and abs(slope(edge)-slope(start_edge)) < 1)):
            temp_result.append(edge)
            start_edge = edge
            m += 1
    return 0




def distance(edge1,edge2):
    x1,y1 = edge1[0]['x'], edge1[0]['y']  #起点
    x11,y11 = edge1[1]['x'], edge1[1]['y'] #终点

    x2,y2 = edge2[0]['x'], edge2[0]['y']  #起点
    x21,y21 = edge2[1]['x'], edge2[1]['y'] #终点

    dis1 = math.sqrt(pow(x11-x2, 2)+pow(y11-y2, 2))
    dis2 = math.sqrt(pow(x11-x21,2)+pow(y11-y21, 2))
    return dis1 if dis1 < dis2 else dis2

#到处shp文件
def export_shp():

    pass

if __name__ == '__main__':
    G = nx.read_gpickle('/home/zyp/network_extract_test/LDC001_images/LDC001_2_2_t_b/LDC001_2_2_t_b_graph_p5_r0.gpickle')
    edgeList = G.edges(data=True)
    node_collection = dict(G.nodes(data=True))
    converted_edgelist = convert_nodeid_to_coordinate(edgeList, node_collection)

    '''
        应用于extension
    '''
    filtered_edge = filterEdge(converted_edgelist)
    sorted_edge_by_conductivity = sorted(filtered_edge, key=lambda edge: edge[3]['conductivity'], reverse=True)
    extension(sorted_edge_by_conductivity, sorted_edge_by_conductivity[0])
    '''
        应用与extension1的部分结束
    '''





    #sorted_edge_by_x_coordinate = sorted(converted_edgelist, key=lambda edge: edge[0]['x'], reverse=True)



    #尝试根据横坐标最左边的作为起点来延伸
    #result = extension4(sorted_edge_by_x_coordinate)

    #将nodeid 转换为坐标之后 避免每次都要传node_collection作为参数
    #edges = G.edge
    #filteredEdge = filterEdge(converted_edgelist)
    #sorted_edge_list = sorted(filteredEdge, key=lambda edge: edge[3]['conductivity'], reverse=True)
    #start_edge = sorted_edge_list[0]
    #result = extension3(sorted_edge_list)

    #start_edge = converted_edgelist[0]
    #sorted_edge_list = extension(converted_edgelist,  start_edge)

    #extension2(sorted_edge_list, start_edge)
