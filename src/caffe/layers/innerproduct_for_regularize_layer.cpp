#include <iostream>
#include <algorithm>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductForRegularizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	InnerProductLayer<Dtype>::LayerSetUp(bottom,top);
}

template <typename Dtype>
void InnerProductForRegularizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	InnerProductLayer<Dtype>::Reshape(bottom,top);

	top[1]->ReshapeLike(*this->blobs_[0]);
}

template <typename Dtype>
void InnerProductForRegularizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	InnerProductLayer<Dtype>::Forward_cpu(bottom,top);

	caffe_copy(top[1]->count(),this->blobs_[0]->cpu_data() ,top[1]->mutable_cpu_data());
}

template <typename Dtype>
void InnerProductForRegularizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	InnerProductLayer<Dtype>::Backward_cpu(top,propagate_down,bottom);

	caffe_add(top[1]->count(), top[1]->cpu_diff(), this->blobs_[0]->cpu_diff(), this->blobs_[0]->mutable_cpu_diff());
}
#ifdef CPU_ONLY
//STUB_GPU(InnerProductForRegularizeLayer);
#endif

INSTANTIATE_CLASS(InnerProductForRegularizeLayer);
REGISTER_LAYER_CLASS(InnerProductForRegularize);
}

