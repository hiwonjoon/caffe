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
void LabelAccuracyWithConfusionLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
  num_label_ = this->layer_param_.accuracy_param().num_label();
}

template <typename Dtype>
void LabelAccuracyWithConfusionLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);

  top_shape.clear();
  top_shape.push_back(num_label_);
  top_shape.push_back(num_label_);
  top[1]->Reshape(top_shape);
}

template <typename Dtype>
void LabelAccuracyWithConfusionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  Dtype* top_confusion = top[1]->mutable_cpu_data();
  caffe_set( top[1]->count(), (Dtype)0., top_confusion );

  int count = 0;
  for (int i = 0; i < bottom[0]->count(); ++i) {
      const int label_value =
          static_cast<int>(bottom_label[i]);
	  const int data_value = 
		  static_cast<int>(bottom_data[i]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_label_);

	  if( label_value == data_value )
		  ++accuracy;
	  top_confusion[label_value * num_label_ + data_value] += (Dtype)1.;
      ++count;
  }

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(LabelAccuracyWithConfusionLayer);
REGISTER_LAYER_CLASS(LabelAccuracyWithConfusion);

}  // namespace caffe
