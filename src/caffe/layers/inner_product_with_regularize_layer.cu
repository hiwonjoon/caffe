#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductWithRegularizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	InnerProductLayer<Dtype>::Forward_gpu(bottom,top);

	vector<Blob<Dtype>*> regu_bot;
	regu_bot.resize(1);
	regu_bot[0] = this->blobs_[0].get();

	vector<Blob<Dtype>*> regu_top;
	regu_top.resize(1);
	regu_top[0] = top[1];

	(static_cast<RegularizeLayer<Dtype> *>(regularize_layer_.get()))->Forward_gpu(regu_bot,regu_top);
  }

template <typename Dtype>
void InnerProductWithRegularizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	InnerProductLayer<Dtype>::Backward_gpu(top,propagate_down,bottom);

	vector<Blob<Dtype>*> regu_bot;
	regu_bot.resize(1);
	regu_bot[0] = this->blobs_[0].get();

	vector<Blob<Dtype>*> regu_top;
	regu_top.resize(1);
	regu_top[0] = top[1];

	vector<bool> regu_prop_down;
	regu_prop_down.resize(1);
	regu_prop_down[0] = propagate_down[1];

	(static_cast<RegularizeLayer<Dtype> *>(regularize_layer_.get()))->Backward_gpu(regu_top,regu_prop_down,regu_bot);
  }

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductWithRegularizeLayer);

}  // namespace caffe
