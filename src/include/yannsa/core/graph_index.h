#ifndef YANNSA_GRAPH_INDEX_H
#define YANNSA_GRAPH_INDEX_H 

#include "yannsa/base/type_definition.h"
#include "yannsa/core/base_index.h"
#include "yannsa/util/heap.h"
#include "yannsa/util/parameter.h"
#include "yannsa/util/base_encoder.h"
#include "yannsa/util/point_pair_distance_table.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <memory>
#include <iostream>
#include <climits>

#include <map>
#include <algorithm>

namespace yannsa {
namespace core {

template <typename PointType, typename DistanceFuncType, typename DistanceType = float>
class GraphIndex : public BaseIndex<PointType, DistanceFuncType, DistanceType> {
  public:
    typedef BaseIndex<PointType, DistanceFuncType, DistanceType> BaseClass;
    typedef typename BaseClass::Dataset Dataset;
    typedef typename BaseClass::DatasetPtr DatasetPtr;
    typedef typename BaseClass::PointVector PointVector;

  private:
    typedef std::unordered_map<IntCode, std::vector<IntIndex> > Bucket2Point; 
    typedef std::shared_ptr<Bucket2Point> Bucket2PointPtr;
    typedef PointDistancePair<IntCode, IntCode> BucketDistancePairItem;
    typedef util::Heap<BucketDistancePairItem> BucketHeap;
    typedef std::unordered_map<IntCode, BucketHeap> BucketKnnGraph;
    typedef std::shared_ptr<BucketKnnGraph> BucketKnnGraphPtr;
    typedef std::vector<IntCode> BucketList;

    typedef PointDistancePair<IntIndex, DistanceType> PointDistancePairItem; 
    struct IndexNode {
      IndexNode(int neighbor_num) : nearest_neighbor(neighbor_num) {}
      util::Heap<PointDistancePairItem> nearest_neighbor;
    };

  public:
    GraphIndex(typename BaseClass::DatasetPtr& dataset_ptr) : BaseClass(dataset_ptr) {}

    void Build(const util::GraphIndexParameter& index_param, 
               util::BaseEncoder<PointType>& encoder); 

    void Clear() {
      index2key_.clear();
      index2neighbor_.clear();
    }

    void SearchKnn(const PointType& query, 
                   int k, 
                   std::vector<std::string>& search_result); 

  private:
    IntCode CalculateHammingDistance(IntCode x, IntCode y) {
      IntCode hamming_dist = 0;
      IntCode xor_result = x ^ y;
      while (xor_result) {
        xor_result &= xor_result-1;
        hamming_dist++;
      }
      return hamming_dist;
    }

    void Encode2Buckets(Bucket2Point& bucket2point, 
                        util::BaseEncoder<PointType>& encoder,
                        int point_neighbor_num); 

    void SplitOneBucket(Bucket2Point& bucket2point, 
                     IntCode cur_bucket, 
                     int max_bucket_size, 
                     int min_bucket_size);

    void SplitBuckets(Bucket2Point& bucket2point, 
                      int max_bucket_size,
                      int min_bucket_size,
                      BucketList& to_split_bucket_list); 

    void MergeOneBucket(Bucket2Point& bucket2point,
                        IntCode cur_bucket,
                        int max_bucket_size,
                        int min_bucket_size,
                        BucketHeap& bucket_neighbor_dist,
                        std::unordered_set<IntCode>& merged_buckets); 

    void MergeBuckets(Bucket2Point& bucket2point, 
                      int max_bucket_size,
                      int min_bucket_size,
                      BucketKnnGraph& to_merge_bucket_knn_graph); 

    void BuildBucketsKnnGraph(BucketList& bucket_list,
                              BucketList& neighbor_bucket_list,
                              int bucket_neighbor_num,
                              BucketKnnGraph& bucket_knn_graph); 

    void BuildPointsKnnGraph(Bucket2Point& bucket2point);

    void GetSplitMergeBucketsList(Bucket2Point& bucket2point, 
                                  BucketList& bucket_list, 
                                  BucketList& to_split_bucket_list, 
                                  BucketList& to_merge_bucket_list,
                                  int max_bucket_size, 
                                  int min_bucket_size); 
    
  private:
    std::vector<std::string> index2key_;
    std::vector<IndexNode> index2neighbor_;
};

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Build(
    const util::GraphIndexParameter& index_param, 
    util::BaseEncoder<PointType>& encoder) {
  Clear();

  // encode
  Bucket2PointPtr bucket2point_ptr(new Bucket2Point()); 
  Encode2Buckets(*bucket2point_ptr, encoder, index_param.point_neighbor_num); 
  
  std::cout << "encode buckets done" << std::endl;

  // get bucket list
  int max_bucket_size = index_param.max_bucket_size;
  int min_bucket_size = index_param.min_bucket_size;
  BucketList bucket_list, to_split_bucket_list, to_merge_bucket_list;
  GetSplitMergeBucketsList(*bucket2point_ptr, bucket_list, to_split_bucket_list, 
                           to_merge_bucket_list, max_bucket_size, min_bucket_size);

  std::cout << "get split merge buckets done" << std::endl;

  // construct need merged bucket knn graph
  BucketKnnGraphPtr to_merge_bucket_knn_graph_ptr(new BucketKnnGraph());
  BuildBucketsKnnGraph(to_merge_bucket_list, bucket_list, 
                       index_param.bucket_neighbor_num, 
                       *to_merge_bucket_knn_graph_ptr);

  std::cout << "build merge buckets knn graph done" << std::endl;

  // split firstly for that too small bucketscan be merged
  SplitBuckets(*bucket2point_ptr, max_bucket_size, min_bucket_size, to_split_bucket_list); 

  std::cout << "split done" << std::endl;

  // merge buckets
  MergeBuckets(*bucket2point_ptr, max_bucket_size, min_bucket_size, *to_merge_bucket_knn_graph_ptr);

  std::cout << "merge done" << std::endl;

  // build point knn graph
  BuildPointsKnnGraph(*bucket2point_ptr);

  // build
  this->have_built_ = true;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::Encode2Buckets(
    Bucket2Point& bucket2point, 
    util::BaseEncoder<PointType>& encoder,
    int point_neighbor_num) { 
  bucket2point.clear();
  typename Dataset::Iterator iter = this->dataset_ptr_->Begin();
  while (iter != this->dataset_ptr_->End()) {
    std::string& key = iter->first;
    PointType& point = iter->second;

    // record point key and index
    IntIndex point_index = index2key_.size();
    index2key_.push_back(key);
    index2neighbor_.push_back(IndexNode(point_neighbor_num));

    // encode point
    IntCode point_code = encoder.Encode(point);
    bucket2point[point_code].push_back(point_index);

    iter++;
  }
  
  /*
  std::unordered_map<IntCode, std::vector<IntIndex> >::iterator it = bucket2point.begin(); 
  for(; it != bucket2point.end(); it++) {
    std::cout << it->first << " : " << it->second.size() << std::endl;
  }
  */
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::GetSplitMergeBucketsList(
    Bucket2Point& bucket2point, BucketList& bucket_list, 
    BucketList& to_split_bucket_list, BucketList& to_merge_bucket_list,
    int max_bucket_size, int min_bucket_size) {
  for (auto& item : bucket2point) {
    bucket_list.push_back(item.first);
    if (item.second.size() > max_bucket_size + min_bucket_size) {
      to_split_bucket_list.push_back(item.first);
    }
    if (item.second.size() < min_bucket_size) {
      to_merge_bucket_list.push_back(item.first);
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitBuckets(
    Bucket2Point& bucket2point, 
    int max_bucket_size,
    int min_bucket_size,
    BucketList& to_split_bucket_list) {
  for (IntCode big_bucket : to_split_bucket_list) {
    SplitOneBucket(bucket2point, big_bucket, max_bucket_size, min_bucket_size);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SplitOneBucket(
    Bucket2Point& bucket2point, 
    IntCode cur_bucket, 
    int max_bucket_size, 
    int min_bucket_size) {
  int high_bits_num = sizeof(IntCode) * CHAR_BIT / 2; 
  int new_bucket_count = 0;
  while (bucket2point[cur_bucket].size() > max_bucket_size + min_bucket_size) {
    new_bucket_count++;
    // avoid overflow
    if (new_bucket_count > (1 << (high_bits_num - 1)) - 1) {
        break;
    }

    IntCode new_bucket = cur_bucket + (new_bucket_count << high_bits_num);
    auto bucket_begin_iter = bucket2point[cur_bucket].end() - max_bucket_size;
    auto bucket_end_iter = bucket2point[cur_bucket].end();
    bucket2point[new_bucket] = std::vector<IntIndex>(bucket_begin_iter, bucket_end_iter);
    bucket2point[cur_bucket].erase(bucket_begin_iter, bucket_end_iter); 
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::MergeOneBucket(
    Bucket2Point& bucket2point,
    IntCode cur_bucket,
    int max_bucket_size,
    int min_bucket_size,
    BucketHeap& bucket_neighbor_dist,
    std::unordered_set<IntCode>& merged_buckets) {
  if (bucket2point[cur_bucket].size() < min_bucket_size) {
    bucket_neighbor_dist.Sort();
    auto bucket_neighbor_iter = bucket_neighbor_dist.Begin();
    for (; bucket_neighbor_iter != bucket_neighbor_dist.End(); bucket_neighbor_iter++) {
      IntCode neighbor_bucket = bucket_neighbor_iter->id;
      // neighbor bucket has been merged
      if (merged_buckets.find(neighbor_bucket) != merged_buckets.end()) {
        continue;
      }
      
      // if not exceed split threshold
      if (bucket2point[neighbor_bucket].size() + bucket2point[cur_bucket].size() 
          <= max_bucket_size + min_bucket_size) {
        bucket2point[cur_bucket].insert(bucket2point[cur_bucket].end(),
                                        bucket2point[neighbor_bucket].begin(),
                                        bucket2point[neighbor_bucket].end());
        merged_buckets.insert(neighbor_bucket);
      }

      // check
      if (bucket2point[cur_bucket].size() > min_bucket_size) {
        break;
      }
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::MergeBuckets(
    Bucket2Point& bucket2point, 
    int max_bucket_size,
    int min_bucket_size,
    BucketKnnGraph& to_merge_bucket_knn_graph) {
  // split and merge
  std::unordered_set<IntCode> merged_buckets;
  int split_threshold = static_cast<int>(max_bucket_size + min_bucket_size);
  auto iter = to_merge_bucket_knn_graph.begin(); 
  for (; iter != to_merge_bucket_knn_graph.end(); iter++) {
    IntCode cur_bucket = iter->first;
    // check whether current bucket has been merged 
    if (merged_buckets.find(cur_bucket) != merged_buckets.end()) {
      continue;
    }

    BucketHeap& bucket_neighbor_dist = iter->second;
    MergeOneBucket(bucket2point, cur_bucket, max_bucket_size, min_bucket_size, 
                bucket_neighbor_dist, merged_buckets); 
  }

  // remove merged buckets
  for (auto bucket_id : merged_buckets) {
    bucket2point.erase(bucket_id);
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildBucketsKnnGraph(
    BucketList& bucket_list,
    BucketList& neighbor_bucket_list,
    int bucket_neighbor_num,
    BucketKnnGraph& bucket_knn_graph) {

  for (IntCode cur_bucket : bucket_list) {
    // avoid tmp heap obj
    bucket_knn_graph.insert(BucketKnnGraph::value_type(cur_bucket, BucketHeap(bucket_neighbor_num)));
    for (IntCode neighbor_bucket : neighbor_bucket_list) {
      if (cur_bucket != neighbor_bucket) {
        // calculate hamming distance
        IntCode hamming_dist = CalculateHammingDistance(cur_bucket, neighbor_bucket);
        bucket_knn_graph[cur_bucket].Insert(BucketDistancePairItem(neighbor_bucket, hamming_dist)); 
      }
    }
  }
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::BuildPointsKnnGraph(
    Bucket2Point& bucket2point) {
  util::PointPairDistanceTable<IntIndex, DistanceType>  point_pair_distance_table;
  DistanceFuncType distance_func;
  auto iter = bucket2point.begin(); 
  for (; iter != bucket2point.end(); iter++) {
    std::vector<IntIndex>& bucket_points = iter->second;
    for (int i = 0; i < bucket_points.size(); i++) {
      IntIndex cur_point = bucket_points[i];
      for (int j = 0; j < bucket_points.size(); j++) {
        if (i != j) {
          IntIndex neighbor_point = bucket_points[j];

          // avoid repeated distance calculation
          DistanceType dist;
          bool dist_exist = point_pair_distance_table.Get(cur_point, neighbor_point, dist);
          if (!dist_exist) {
            dist = distance_func(this->dataset_ptr_->Get(index2key_[cur_point]),
                                 this->dataset_ptr_->Get(index2key_[neighbor_point])); 
            point_pair_distance_table.Insert(cur_point, neighbor_point, dist);
          }

          index2neighbor_[cur_point].nearest_neighbor.Insert(PointDistancePairItem(neighbor_point, dist));
        }
      }
    }
  }

  std::cout << "build point knn graph done" << std::endl;

  std::map<IntIndex, int> point_neighbored_count;
  for (auto& index_node : index2neighbor_) {
    auto& nearest_neighbor = index_node.nearest_neighbor;
    for (auto iter = nearest_neighbor.Begin(); iter != nearest_neighbor.End(); iter++) {
      auto cur_count = point_neighbored_count.find(iter->id);
      if (cur_count == point_neighbored_count.end()) {
        point_neighbored_count[iter->id] = 1;
      }
      else {
        cur_count->second += 1;
      }
    }
  }

  std::vector<int> point_count_vector;
  for (auto& count_info : point_neighbored_count) {
    point_count_vector.push_back(count_info.second);
  }
  std::sort(point_count_vector.begin(), point_count_vector.end());
  std::reverse(point_count_vector.begin(), point_count_vector.end());

  for (int i = 0; i < 100; i++) {
    std::cout << point_count_vector[i] << "\t";
  }

  std::cout << std::endl;
}

template <typename PointType, typename DistanceFuncType, typename DistanceType>
void GraphIndex<PointType, DistanceFuncType, DistanceType>::SearchKnn(
    const PointType& query, 
    int k, 
    std::vector<std::string>& search_result) {
  search_result.clear();
  // Init some points, search from these points
}

} // namespace core 
} // namespace yannsa

#endif
