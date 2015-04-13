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
void RegularizeResultLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	
}
template <typename Dtype>
void RegularizeResultLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void RegularizeResultLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	top[0]->mutable_cpu_data()[0] = bottom[0]->cpu_data()[0];
}
template <typename Dtype>
void RegularizeResultLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if( propagate_down[0] ) {
		bottom[0]->mutable_cpu_diff()[0] = top[0]->cpu_diff()[0];
	}
}
INSTANTIATE_CLASS(RegularizeResultLayer);
REGISTER_LAYER_CLASS(RegularizeResult);
}
