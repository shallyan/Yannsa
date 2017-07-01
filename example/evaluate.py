#coding=utf-8

import sys

if len(sys.argv) < 4:
  print 'evaluate -graph -ground_truth -k'
  sys.exit()
  
graph_path = sys.argv[1]
real_graph_path = sys.argv[2]
k = int(sys.argv[3])

def ReadGraph(graph_path):
  graph_list = list()
  with open(graph_path) as f:
    for line in f:
      fields = line.strip().split()
      neighbor = set(fields)
      if len(neighbor) != len(fields):
        print fields
        print 'knn not unique'
      graph_list.append(neighbor)
  return graph_list

result_graph = ReadGraph(graph_path)
real_graph = ReadGraph(real_graph_path)

hit = 0
for real, result in zip(real_graph, result_graph):
  cur_hit = len(result & real)
  print cur_hit
  hit += cur_hit

print 'Average precision: ', hit * 1.0 / (len(real_graph) * k)
