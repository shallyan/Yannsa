#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "yannsa/util/parameter.h"
#include "yannsa/wrapper/index_helper.h"
#include <iostream>
#include <set>
#include <fstream>
#include <utility>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace std;
using namespace yannsa;
using namespace yannsa::core;
using namespace yannsa::util;
using namespace yannsa::wrapper;

template<typename DistanceType>
class Index { 
  private:
    typedef GraphIndex<PointVector<float>, DistanceType, float> GraphIndexType;

  public:
    Index(bool is_normalize=false) : is_normalize_(is_normalize), 
          dataset_ptr_(new Dataset<float>()), 
          graph_index_ptr_(new GraphIndexType(dataset_ptr_)) { 
    }

    void InitData(const string file_path) {
      ifstream in_file(file_path, ios::binary);

      int point_dim = 0;
      in_file.read(reinterpret_cast<char*>(&point_dim), sizeof(int));

      in_file.seekg(0, ios::end);
      IntIndex point_num = in_file.tellg() / (4 + point_dim * 4);

      cout << file_path << " has " << point_num << " points and " << point_dim << " dims" << endl;
      in_file.seekg(0, ios::beg);
      for (IntIndex point_id = 0; point_id < point_num; point_id++) {
        in_file.seekg(4, ios::cur);
        PointVector<float> point(point_dim);
        float value = 0;
        for (int d = 0; d < point_dim; d++) {
          in_file.read(reinterpret_cast<char*>(&value), sizeof(float));
          point[d] = value;
        }

        stringstream key_str;
        key_str << point_id;
        string key;
        key_str >> key;
        dataset_ptr_->insert(key, point);
      }

      in_file.close();
      cout << "create dataset done, data num: " 
           << dataset_ptr_->size() << endl;

    }

    void Build(int k, int join_k, int refine_iter_num, float lambda_value) {
      int least_data_size = min(join_k, k) * 2;
      if (dataset_ptr_->size() < least_data_size) {
        throw IndexBuildError("Indiced data size is too small: " + to_string(dataset_ptr_->size()));
      }

      util::GraphIndexParameter index_param;
      index_param.k = k;
      index_param.join_k = join_k;
      index_param.refine_iter_num = refine_iter_num;
      index_param.lambda = lambda_value;

      graph_index_ptr_->Build(index_param);
    }

    void SaveIndex(const string index_save_path) {
      graph_index_ptr_->SaveIndex(index_save_path);
    }

    void LoadIndex(const string index_save_path) {
      if (dataset_ptr_->size() == 0) {
        throw IndexBuildError("Vectors have not been loaded!");
      }

      graph_index_ptr_->LoadIndex(index_save_path);
    }

    vector<string> SearchKnn(Eigen::Ref<PointVector<float> > query, int K, int search_K) {
      GraphSearchParameter search_param;
      search_param.K = K;
      search_param.search_K = search_K;

      vector<string> ret;
      graph_index_ptr_->SearchKnn(query, search_param, ret);
      return ret;
    }

    void AddPoint(const string key, Eigen::Ref<PointVector<float> > point, int search_K) {
      graph_index_ptr_->AddNewPoint(key, point, search_K);
    }

  private:
    bool is_normalize_;
    DatasetPtr<float> dataset_ptr_;
    shared_ptr<GraphIndexType> graph_index_ptr_;
};

PYBIND11_MODULE(yannsa, m)
{
  // optional module docstring
  m.doc() = "Yannsa Python Binding";

  py::class_<Index<EuclideanDistance<float> > >(m, "EuclideanIndex")
    .def(py::init<bool>(), py::arg("is_normalize")=false)
    .def("init_data", &Index<EuclideanDistance<float> >::InitData, py::arg("file_path"))
    .def("build", &Index<EuclideanDistance<float> >::Build, py::arg("k"), py::arg("join_k"), py::arg("refine_iter_num"), py::arg("lambda_value"))
    .def("save_index", &Index<EuclideanDistance<float> >::SaveIndex, py::arg("index_save_path"))
    .def("load_index", &Index<EuclideanDistance<float> >::LoadIndex, py::arg("index_save_path"))
    .def("search_knn", &Index<EuclideanDistance<float> >::SearchKnn, py::arg("query"), py::arg("K"), py::arg("search_K"))
    .def("add_point", &Index<EuclideanDistance<float> >::AddPoint, py::arg("key"), py::arg("point"), py::arg("search_K"));

  py::class_<Index<DotDistance<float> > >(m, "CosineIndex")
    .def(py::init<bool>(), py::arg("is_normalize")=true)
    .def("init_data", &Index<DotDistance<float> >::InitData, py::arg("file_path"))
    .def("build", &Index<DotDistance<float> >::Build, py::arg("k"), py::arg("join_k"), py::arg("refine_iter_num"), py::arg("lambda_value"))
    .def("save_index", &Index<DotDistance<float> >::SaveIndex, py::arg("index_save_path"))
    .def("load_index", &Index<DotDistance<float> >::LoadIndex, py::arg("index_save_path"))
    .def("search_knn", &Index<DotDistance<float> >::SearchKnn, py::arg("query"), py::arg("K"), py::arg("search_K"))
    .def("add_point", &Index<DotDistance<float> >::AddPoint, py::arg("key"), py::arg("point"), py::arg("search_K"));
}
