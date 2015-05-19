#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TopLabelLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void TopLabelLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  vector<int> top_shape;  // Accuracy is a scalar; 0 axes.
  top_shape.push_back(outer_num_*inner_num_);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void TopLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  //vector<Dtype> maxval(top_k_+1);
  //vector<int> max_id(top_k_+1);
  //int count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + 1,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
	  top[0]->mutable_cpu_data()[inner_num_*i+j] = bottom_data_vector[0].second;
      // check if true label is in top k predictions
      //for (int k = 0; k < top_k_; k++) {
      //  if (bottom_data_vector[k].second == label_value) {
      //    ++accuracy;
      //    break;
      //  }
      //}
      //++count;
    }
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  //top[0]->mutable_cpu_data()[0] = accuracy / count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(TopLabelLayer);
REGISTER_LAYER_CLASS(TopLabel);
}
