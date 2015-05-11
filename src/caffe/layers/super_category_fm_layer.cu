#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SuperCategoryFMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	for(int i = 0; i < depth_; ++i) {
		for(int j = 0; j < node_num_per_level_[i]; ++j) {
			if( i == depth_-1 ) {
				for(int m = 0; m < M_; ++m) {
					caffe_copy(H_*W_, &bottom[0]->gpu_data()[bottom[0]->offset(m,j)],&temp_blobs[i][j]->mutable_gpu_data()[temp_blobs[i][j]->offset(m)]);
				}
			}
			else
				caffe_gpu_set(temp_blobs[i][j]->count(), (Dtype)0., temp_blobs[i][j]->mutable_gpu_data());
		}
	}

	for( int i = depth_-2; i >= 0; --i ) {
		int base_idx = base_index_per_level_[i];
		for(int j = 0; j < node_num_per_level_[i]; ++j) {
			Tree * node = serialized_tree_[base_idx + j];
			const std::vector<shared_ptr<Tree> >* children = node->GetChildren();

			shared_ptr<Blob<Dtype> > tops = temp_blobs[i][node->GetLabel()];
			Dtype * top_data = tops->mutable_gpu_data();

			for(std::vector<shared_ptr<Tree> >::const_iterator it = children->begin(); it != children->end(); ++it) {
				shared_ptr<Blob<Dtype> > bottoms = temp_blobs[i+1][(*it)->GetLabel()];
				const Dtype * bottom_data = bottoms->gpu_data();
				caffe_gpu_axpy(M_*H_*W_,(Dtype)(1.),bottom_data,top_data);
			}
			caffe_gpu_scal(M_*H_*W_,(Dtype)(1./children->size()),top_data);
		}
	}

	for(int m = 0; m < M_; ++m) {
		for(int i = 0; i < depth_; ++i) {
			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j) {
				Tree * node = serialized_tree_[base_idx + j];
				caffe_copy(H_*W_,&temp_blobs[i][node->GetLabel()]->gpu_data()[temp_blobs[i][node->GetLabel()]->offset(m)],&top[i]->mutable_gpu_data()[top[i]->offset(m,node->GetLabel())]);
			}
		}
	}
}
template <typename Dtype>
void SuperCategoryFMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	if( propagate_down[0] == false )
		return;

	for(int m = 0; m < M_; ++m) {
		for(int i = 0; i < depth_; ++i) {
			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j) {
				Tree * node = serialized_tree_[base_idx + j];
				caffe_copy(H_*W_,&top[i]->gpu_diff()[top[i]->offset(m,node->GetLabel())],&temp_blobs[i][node->GetLabel()]->mutable_gpu_diff()[temp_blobs[i][node->GetLabel()]->offset(m)]);
			}
		}
	}

	for( int i = 0; i < depth_-1; ++i ) {
		int base_idx = base_index_per_level_[i];
		for(int j = 0; j < node_num_per_level_[i]; ++j) {
			Tree * node = serialized_tree_[base_idx + j];
			const std::vector<shared_ptr<Tree> >* children = node->GetChildren();

			shared_ptr<Blob<Dtype> > tops = temp_blobs[i][node->GetLabel()];
			const Dtype * top_diff = tops->gpu_diff();
			for(std::vector<shared_ptr<Tree> >::const_iterator  it = children->begin(); it != children->end(); ++it) {
				shared_ptr<Blob<Dtype> > bottoms = temp_blobs[i+1][(*it)->GetLabel()];
				Dtype * bottom_diff = bottoms->mutable_gpu_diff();

				caffe_gpu_axpy(M_*H_*W_,(Dtype)(1./children->size()),top_diff,bottom_diff);	
			}

		}
	}

	for(int m = 0; m < M_; ++m) {
		for(int i = 0; i < depth_; ++i) {
			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j) {
				Tree * node = serialized_tree_[base_idx + j];
				caffe_copy(H_*W_,&temp_blobs[i][node->GetLabel()]->gpu_diff()[temp_blobs[i][node->GetLabel()]->offset(m)],&top[i]->mutable_gpu_diff()[top[i]->offset(m,node->GetLabel())]);
			}
		}
	}
	caffe_copy(bottom[0]->count(), top[depth_-1]->gpu_diff(), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(SuperCategoryFMLayer);

}  // namespace caffe
