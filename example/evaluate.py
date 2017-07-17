#coding=utf-8

import sys

if len(sys.argv) < 4:
  print 'evaluate -graph -ground_truth -k'
  sys.exit()
  
graph_path = sys.argv[1]
real_graph_path = sys.argv[2]
k = int(sys.argv[3])

def ReadGraph(graph_path):
  graph_dict = dict()
  with open(graph_path) as f:
    for line in f:
      fields = line.strip().split()
      point = fields[0]
      knn = fields[1:k+1]
      neighbor = set(knn)
      """
      if len(neighbor) != len(knn):
        print fields
        print 'knn not unique'
      """
      graph_dict[point] = neighbor
  return graph_dict

result_graph = ReadGraph(graph_path)
real_graph = ReadGraph(real_graph_path)

hit = 0
for point, real_neighbor in real_graph.iteritems():
  result_neighbor = result_graph[point]
  cur_hit = len(result_neighbor & real_neighbor)
  hit += cur_hit

print 'Average precision: ', hit * 1.0 / (len(real_graph) * k)
