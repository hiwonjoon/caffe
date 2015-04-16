#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void RegularizeLayer<Dtype>::Forward_gpu_impl(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	temp_.CopyFrom(*bottom[0],false,true);

	const int count = bottom[0]->count();
	Dtype * temp_data = temp_.mutable_gpu_data();
	caffe_gpu_powx(count,temp_data,(Dtype)2,temp_data);

	//for(int i = 0 ; i < bottom[0]->shape(0); ++i) {
	//	for (int j = 0; j < bottom[0]->shape(1); ++j) {
	//		std::cout << temp_data[i*bottom[0]->shape(1)+j] << " ";
	//	}
	//	std::cout << std::endl;
	//}

	const Dtype * tree_map_data = tree_map_.gpu_data();
	Dtype* gv_map_data = gv_map_.mutable_gpu_data();		

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, node_, K_, N_, (Dtype)1., tree_map_data, temp_.gpu_data(), (Dtype)0., gv_map_data);

	//for(int i = 0; i < node_;  ++i) {
	//	for(int j = 0; j < K_; ++j) {
	//		std::cout << gv_map_data[i*K_+j] << " ";
	//	}
	//	std::cout << std::endl;
	//}

	caffe_gpu_powx( gv_map_.count(), gv_map_.gpu_data(), (Dtype)0.5, gv_map_.mutable_gpu_data());
	
	//for(int i = 0; i < node_;  ++i) {
	//	for(int j = 0; j < K_; ++j) {
	//		std::cout << gv_map_data[i*K_+j] << " ";
	//	}
	//	std::cout << std::endl;
	//}

	const Dtype * g_agg_data = g_agg_.gpu_data();

	//for(int i = 0; i < node_; i++)
	//{
	//	std::cout << g_agg_data[i] << " "; 
	//}
	//std::cout << std::endl;

	Blob<Dtype> temp2(1, K_, 1, 1);
	Dtype* temp2_data = temp2.mutable_gpu_data();
	caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, K_, node_, (Dtype)1., g_agg_data, gv_map_.gpu_data(), (Dtype)0., temp2_data);

	
	//for(int i = 0; i < K_; i++) {
	//	std::cout << temp2_data[i] << " ";
	//}
	//std::cout << std::endl;

	top[0]->mutable_cpu_data()[0] = temp2.asum_data();
}

template <typename Dtype>
void RegularizeLayer<Dtype>::Backward_gpu_impl(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom, Blob<Dtype>& diff_g, Blob<Dtype>& temp2) {
	const Dtype * top_diff = top[0]->cpu_diff();
	temp_.Reshape(node_,K_,1,1);

	//const Dtype * diff_g_data = diff_g.gpu_data();
	//for(int i = 0; i < internal_node_; ++i) {
	//	for(int j = 0; j < node_; ++j) {
	//		std::cout << diff_g_data[i* node_ + j] << " " ;
	//	}
	//	std::cout << std::endl;
	//}
	
	caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,internal_node_,K_,node_, *top_diff, diff_g.gpu_data(),gv_map_.gpu_data(), (Dtype)0., temp_.mutable_gpu_data()); 

	//gradient with bottom data
	if( propagate_down[0] )
	{
		int count = gv_map_.count();
		Dtype* gv_map_data = gv_map_.mutable_gpu_data();
		caffe_gpu_powx(count, gv_map_data, (Dtype)-1, gv_map_data);

		Dtype * temp2_data = temp2.mutable_gpu_data();
		//for(int i = 0; i < node_; ++i)
		//{
		//	for(int j = 0; j < N_; ++j)
		//		std::cout << temp_data[i*N_+j] << " ";
		//	std::cout << std::endl;
		//}
		
		temp_.Reshape(N_,K_,1,1);
		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, node_, *top_diff, temp2_data, gv_map_data, (Dtype)0., temp_.mutable_gpu_diff());
		caffe_gpu_mul<Dtype>(temp_.count(), bottom[0]->gpu_data(), temp_.gpu_diff(), temp_.mutable_gpu_diff());
		
		//caffe_add<Dtype>(temp_.count(), temp_.gpu_diff(), bottom[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
		caffe_gpu_axpy(temp_.count(), Dtype(1), temp_.gpu_diff(), bottom[0]->mutable_gpu_diff());
	}

}

  template void RegularizeLayer<float>::Forward_gpu_impl( 
      const std::vector<Blob<float>*>& bottom, 
      const std::vector<Blob<float>*>& top); 
  template void RegularizeLayer<double>::Forward_gpu_impl( 
      const std::vector<Blob<double>*>& bottom, 
      const std::vector<Blob<double>*>& top);

  template void RegularizeLayer<float>::Backward_gpu_impl( 
      const std::vector<Blob<float>*>& top, 
      const std::vector<bool>& propagate_down, 
      const std::vector<Blob<float>*>& bottom,
	  Blob<float>& diff_g,
	  Blob<float>& temp2);
  template void RegularizeLayer<double>::Backward_gpu_impl( 
      const std::vector<Blob<double>*>& top, 
      const std::vector<bool>& propagate_down, 
      const std::vector<Blob<double>*>& bottom,
	  Blob<double>& diff_g,
	  Blob<double>& temp2);

}  // namespace caffe
