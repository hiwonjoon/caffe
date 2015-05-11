#include <vector>
#include <limits>
#include <queue>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
//Layer Implementation
template <typename Dtype>
void SuperCategoryFMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const SuperCategoryParameter super_param = this->layer_param_.super_category_param();

	Tree::MakeTree(&root_, &super_param.root());
	depth_ = root_.Depth() - 1;
	root_.MakeBalance(depth_);
	Tree::GiveIndex(&root_, serialized_tree_);
	Tree::GetNodeNumPerLevelAndGiveLabel(node_num_per_level_, base_index_per_level_, &this->root_,serialized_tree_,label_to_index_);

	M_ = bottom[0]->shape(0);
	N_ = bottom[0]->shape(1);
	H_ = bottom[0]->shape(2);
	W_ = bottom[0]->shape(3);
}

template <typename Dtype>
void SuperCategoryFMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(top.size(), depth_);

	for( int i = 0; i < depth_; ++i) {
		std::vector<int> shape;
		shape.push_back(M_);
		shape.push_back(node_num_per_level_[i]);
		shape.push_back(H_);
		shape.push_back(W_);
		top[i]->Reshape(shape); // Top for output data
	}

}

template <typename Dtype>
void SuperCategoryFMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[depth_-1]->mutable_cpu_data());
	for(int i = 0; i < depth_-1; ++i)
		caffe_set(top[i]->count(), (Dtype)0., top[i]->mutable_cpu_data());

	for(int m = 0; m < M_; ++m) {
		for( int i = depth_-2; i >= 0; --i ) {
			Blob<Dtype> * tops = top[i];
			Blob<Dtype> * bottoms = top[i+1];

			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j) {
				Tree * node = serialized_tree_[base_idx + j];
				const std::vector<shared_ptr<Tree> >* children = node->GetChildren();

				Dtype * top_data = &tops->mutable_cpu_data()[tops->offset(m,node->GetLabel())];

				for(auto it = children->cbegin(); it != children->cend(); ++it) {
					int offset = bottoms->offset(m,(*it)->GetLabel());
					const Dtype * bottom_data = &bottoms->cpu_data()[offset];
					caffe_axpy(H_*W_,(Dtype)(1.),bottom_data,top_data);
				}

				caffe_scal(H_*W_,(Dtype)(1./children->size()),top_data);
			}
		}
	}
}

template <typename Dtype>
void SuperCategoryFMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
	if( propagate_down[0] == false )
		return;

	for(int m = 0; m < M_; ++m) {
		for( int i = 0; i < depth_-1; ++i ) {
			Blob<Dtype> * tops = top[i];
			Blob<Dtype> * bottoms = top[i+1];

			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j) {
				Tree * node = serialized_tree_[base_idx + j];
				const std::vector<shared_ptr<Tree> >* children = node->GetChildren();
				const Dtype * top_diff = &tops->cpu_diff()[tops->offset(m,node->GetLabel())];
				for(auto it = children->cbegin(); it != children->cend(); ++it) {
					int offset = bottoms->offset(m,(*it)->GetLabel());
					Dtype * bottom_diff = &bottoms->mutable_cpu_diff()[offset];

					caffe_axpy(H_*W_,(Dtype)(1./children->size()),top_diff,bottom_diff);	
				}

			}
		}
	}
	caffe_copy(bottom[0]->count(), top[depth_-1]->cpu_diff(), bottom[0]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(SuperCategoryFMLayer);
#endif

INSTANTIATE_CLASS(SuperCategoryFMLayer);

REGISTER_LAYER_CLASS(SuperCategoryFM);
}  // namespace caffe

