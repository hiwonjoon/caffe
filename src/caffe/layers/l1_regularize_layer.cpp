#include <iostream>
#include <algorithm>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void L1RegularizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//bottom_ = bottom;
	N_ = bottom[0]->shape(0);
	K_ = bottom[0]->shape(1);

}

template <typename Dtype>
void L1RegularizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	N_ = bottom[0]->shape(0);
	const int new_K = bottom[0]->shape(1);
	CHECK_EQ(K_,new_K) << "input size is not compatible";

	std::vector<int> shape;
	shape.push_back(1);
	top[0]->Reshape(shape);

	shape.clear();
	shape.push_back(N_);
	shape.push_back(K_);
	temp_.Reshape(shape);
}

template <typename Dtype>
void L1RegularizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const int count = bottom[0]->count();
	caffe_cpu_sign(count, bottom[0]->cpu_data(), temp_.mutable_cpu_data());
	caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, 1, count, (Dtype)1., temp_.cpu_data(), bottom[0]->cpu_data(), (Dtype)0., top[0]->mutable_cpu_data());
	*top[0]->mutable_cpu_data() /= N_;
}

template <typename Dtype>
void L1RegularizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const int count = bottom[0]->count();
	if( propagate_down[0] ) {
		caffe_axpy(count, top[0]->cpu_diff()[0] / N_, temp_.mutable_cpu_data(),bottom[0]->mutable_cpu_diff());
	}
}


#ifdef CPU_ONLY
STUB_GPU(L1RegularizeLayer);
#endif

INSTANTIATE_CLASS(L1RegularizeLayer);
REGISTER_LAYER_CLASS(L1Regularize);
}
