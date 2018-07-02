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

    void InitDataInFVecs(const string file_path) {
      ifstream in_file(file_path, ios::binary);

      int point_dim = 0;
      in_file.read(reinterpret_cast<char*>(&point_dim), sizeof(int));

      in_file.seekg(0, ios::end);
      IntIndex point_num = in_file.tellg() / (4 + point_dim * 4);

      cout << file_path << " has " << point_num << " points and " << point_dim << " dims" << endl;
      dataset_ptr_->reserve(point_num);

      in_file.seekg(0, ios::beg);
      for (IntIndex point_id = 0; point_id < point_num; point_id++) {
        in_file.seekg(4, ios::cur);
        PointVector<float> point(point_dim);
        float value = 0;
        for (int d = 0; d < point_dim; d++) {
          in_file.read(reinterpret_cast<char*>(&value), sizeof(float));
          point[d] = value;
        }

        if (is_normalize_) {
          point.normalize();
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

    void InitDataInEmbedding(const string file_path) {
      ifstream in_file(file_path.c_str());

      string buff;
      //read word and dim number
      int vec_num = 0, vec_dim = 0;
      getline(in_file, buff);
      stringstream count_stream(buff);
      count_stream >> vec_num >> vec_dim; 
      cout << "vec num: " << vec_num << "\t" 
           << "vec dim: " << vec_dim << endl;

      while (getline(in_file, buff)) {
        stringstream one_word_vec_stream(buff);

        //read word firstly
        string word;
        one_word_vec_stream >> word;

        //then read value
        PointVector<float> point(vec_dim);
        double value = 0.0;
        int dim_count = 0;
        while (one_word_vec_stream >> value) { 
          point[dim_count++] = value;
        }

        if (is_normalize_) {
          point.normalize();
        }

        dataset_ptr_->insert(word, point);
      }

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
    .def("init_data_fvecs", &Index<EuclideanDistance<float> >::InitDataInFVecs, py::arg("file_path"))
    .def("init_data_embedding", &Index<EuclideanDistance<float> >::InitDataInEmbedding, py::arg("file_path"))
    .def("build", &Index<EuclideanDistance<float> >::Build, py::arg("k"), py::arg("join_k"), py::arg("refine_iter_num"), py::arg("lambda_value"))
    .def("save_index", &Index<EuclideanDistance<float> >::SaveIndex, py::arg("index_save_path"))
    .def("load_index", &Index<EuclideanDistance<float> >::LoadIndex, py::arg("index_save_path"))
    .def("search_knn", &Index<EuclideanDistance<float> >::SearchKnn, py::arg("query"), py::arg("K"), py::arg("search_K"))
    .def("add_point", &Index<EuclideanDistance<float> >::AddPoint, py::arg("key"), py::arg("point"), py::arg("search_K"));

  py::class_<Index<DotDistance<float> > >(m, "CosineIndex")
    .def(py::init<bool>(), py::arg("is_normalize")=true)
    .def("init_data_fvecs", &Index<DotDistance<float> >::InitDataInFVecs, py::arg("file_path"))
    .def("init_data_embedding", &Index<DotDistance<float> >::InitDataInEmbedding, py::arg("file_path"))
    .def("build", &Index<DotDistance<float> >::Build, py::arg("k"), py::arg("join_k"), py::arg("refine_iter_num"), py::arg("lambda_value"))
    .def("save_index", &Index<DotDistance<float> >::SaveIndex, py::arg("index_save_path"))
    .def("load_index", &Index<DotDistance<float> >::LoadIndex, py::arg("index_save_path"))
    .def("search_knn", &Index<DotDistance<float> >::SearchKnn, py::arg("query"), py::arg("K"), py::arg("search_K"))
    .def("add_point", &Index<DotDistance<float> >::AddPoint, py::arg("key"), py::arg("point"), py::arg("search_K"));
}
