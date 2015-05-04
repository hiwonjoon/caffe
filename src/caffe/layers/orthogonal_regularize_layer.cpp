#include <iostream>
#include <limits>
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
void OrthogonalRegularizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	N_lower_ = bottom[0]->shape(0);
	N_upper_ = bottom[1]->shape(0);

	CHECK_EQ(bottom[0]->shape(1),bottom[1]->shape(1));
	K_ = bottom[0]->shape(1);
}

template <typename Dtype>
void OrthogonalRegularizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(N_lower_,bottom[0]->shape(0));
	CHECK_EQ(N_upper_,bottom[1]->shape(0));
	CHECK_EQ(bottom[0]->shape(1),bottom[1]->shape(1));
	const int new_K = bottom[0]->shape(1);
	CHECK_EQ(K_,new_K) << "input size is not compatible";

	std::vector<int> shape;
	shape.push_back(1);
	top[0]->Reshape(shape);


	shape.clear();
	shape.push_back(N_upper_);
	shape.push_back(N_lower_);
	temp_.Reshape(shape);

	shape.clear();
	shape.push_back(N_lower_);
	shape.push_back(K_);
	temp2_.Reshape(shape);

	shape.clear();
	shape.push_back(K_);
	temp3_.Reshape(shape);
}

template <typename Dtype>
void OrthogonalRegularizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Dtype * loss = top[0]->mutable_cpu_data();
	*loss = 0;
	for(int i = 0; i < N_upper_; ++i) {
		const Dtype * upper_w_vector = &bottom[1]->cpu_data()[i*K_];
		Dtype * temp_data = &temp_.mutable_cpu_data()[i*N_lower_];
		for(int j = 0; j < N_lower_; ++j) {
			caffe_copy(K_, upper_w_vector, &temp2_.mutable_cpu_data()[K_*j]);
		}

		caffe_sub( temp2_.count(), temp2_.cpu_data(), bottom[0]->cpu_data(), temp2_.mutable_cpu_data() );

		caffe_cpu_gemm(CblasNoTrans, CblasTrans, 1, N_lower_, K_, (Dtype)1., upper_w_vector, temp2_.cpu_data(), (Dtype)0., temp_data);

	}

	const Dtype * temp_data = temp_.cpu_data();
	for(int j = 0; j < N_lower_; ++j)
	{
		Dtype min = std::numeric_limits<Dtype>::max();
		for(int i = 0; i < N_upper_; ++i)
		{
			if( min > temp_data[i * N_lower_ + j] )
				min = temp_data[i * N_lower_ + j];
		}
		*loss += min;
	}
}

template <typename Dtype>
void OrthogonalRegularizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Dtype top_diff = *top[0]->cpu_diff();

	const Dtype * temp_data = temp_.cpu_data();
	for(int j = 0; j < N_lower_; ++j)
	{
		int i_ptr = 0;
		Dtype min = std::numeric_limits<Dtype>::max();
		for(int i = 0; i < N_upper_; ++i)
		{
			if( min > temp_data[i * N_lower_ + j] )
			{
				min = temp_data[i * N_lower_ + j];
				i_ptr = i;
			}
		}
		caffe_cpu_scale( K_, -1 * top_diff, &bottom[1]->cpu_data()[K_*i_ptr], &temp2_.mutable_cpu_diff()[K_*j]);

		if( propagate_down[1] ) {
			caffe_copy(K_, &bottom[0]->cpu_data()[K_*j], temp3_.mutable_cpu_data());
			caffe_cpu_axpby( K_, (Dtype)2, &bottom[1]->cpu_data()[K_*i_ptr], (Dtype)-1, temp3_.mutable_cpu_data());
			caffe_cpu_axpby( K_, top_diff, temp3_.cpu_data(), (Dtype)1, &bottom[1]->mutable_cpu_diff()[K_*i_ptr]);
		}
	}

	if( propagate_down[0] ) {
		caffe_add( bottom[0]->count(), temp2_.cpu_diff(), bottom[0]->cpu_diff(), bottom[0]->mutable_cpu_diff() );
	}
}


#ifdef CPU_ONLY
STUB_GPU(OrthogonalRegularizeLayer);
#endif

INSTANTIATE_CLASS(OrthogonalRegularizeLayer);
REGISTER_LAYER_CLASS(OrthogonalRegularize);
}
