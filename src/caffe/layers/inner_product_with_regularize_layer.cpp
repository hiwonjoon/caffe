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
void InnerProductWithRegularizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	InnerProductLayer<Dtype>::LayerSetUp(bottom,top);
	regularize_layer_.reset(new RegularizeLayer<Dtype>(this->layer_param_));
	
	vector<Blob<Dtype>*> regu_bot;
	regu_bot.resize(1);
	regu_bot[0] = this->blobs_[0].get();

	vector<Blob<Dtype>*> regu_top;
	regu_top.resize(1);
	regu_top[0] = top[1];

	regularize_layer_->LayerSetUp(regu_bot,regu_top);

	//blob sharing
	this->blobs_.insert(this->blobs_.end(), regularize_layer_->blobs().begin(), regularize_layer_->blobs().end());
	//propagate down parameter sharing
	this->param_propagate_down_.insert(this->param_propagate_down_.end(), static_cast<RegularizeLayer<Dtype>*>(regularize_layer_.get())->param_propagate_down_.begin(), static_cast<RegularizeLayer<Dtype>*>(regularize_layer_.get())->param_propagate_down_.end());
}

template <typename Dtype>
void InnerProductWithRegularizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	InnerProductLayer<Dtype>::Reshape(bottom,top);

	vector<Blob<Dtype>*> regu_bot;
	regu_bot.resize(1);
	regu_bot[0] = this->blobs_[0].get();

	vector<Blob<Dtype>*> regu_top;
	regu_top.resize(1);
	regu_top[0] = top[1];

	regularize_layer_->Reshape(regu_bot, regu_top);
}

template <typename Dtype>
void InnerProductWithRegularizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	InnerProductLayer<Dtype>::Forward_cpu(bottom,top);

	vector<Blob<Dtype>*> regu_bot;
	regu_bot.resize(1);
	regu_bot[0] = this->blobs_[0].get();

	vector<Blob<Dtype>*> regu_top;
	regu_top.resize(1);
	regu_top[0] = top[1];

	(static_cast<RegularizeLayer<Dtype> *>(regularize_layer_.get()))->Forward_cpu(regu_bot,regu_top);
}

template <typename Dtype>
void InnerProductWithRegularizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	InnerProductLayer<Dtype>::Backward_cpu(top,propagate_down,bottom);

	vector<Blob<Dtype>*> regu_bot;
	regu_bot.resize(1);
	regu_bot[0] = this->blobs_[0].get();

	vector<Blob<Dtype>*> regu_top;
	regu_top.resize(1);
	regu_top[0] = top[1];

	vector<bool> regu_prop_down;
	regu_prop_down.resize(1);
	regu_prop_down[0] = propagate_down[1];

	(static_cast<RegularizeLayer<Dtype> *>(regularize_layer_.get()))->Backward_cpu(regu_top,regu_prop_down,regu_bot);
}
#ifdef CPU_ONLY
//STUB_GPU(InnerProductWithRegularizeLayer);
#endif

INSTANTIATE_CLASS(InnerProductWithRegularizeLayer);
REGISTER_LAYER_CLASS(InnerProductWithRegularize);
}
