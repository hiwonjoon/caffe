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
template <typename Dtype>
void SuperCategoryInverseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const SuperCategoryParameter super_param = this->layer_param_.super_category_param();

	Tree::MakeTree(&root_, &super_param.root());
	depth_ = root_.Depth() -1 ;
	root_.MakeBalance(depth_);
	Tree::GiveIndex(&root_, serialized_tree_);
	Tree::GetNodeNumPerLevelAndGiveLabel(node_num_per_level_, base_index_per_level_, &this->root_,serialized_tree_,label_to_index_);

	M_ = bottom[0]->shape(0);
	//N_ = bottom[1]->shape(1);

	CHECK_EQ(bottom.size(), this->depth_ * 2);
	for(int i = 0; i < depth_; ++i) {
		CHECK_EQ(M_,bottom[i*2]->shape(0));
		CHECK_EQ(bottom[i*2]->shape(1),node_num_per_level_[i]); 
		//CHECK_EQ(bottom[i*2+1]->shape(0),node_num_per_level_[i]);
		//CHECK_EQ(bottom[i*2+1]->shape(1),N_);
	}
}

template <typename Dtype>
void SuperCategoryInverseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(top.size(), this->depth_ );
	for(int i = 0; i < depth_; ++i)
		top[i]->ReshapeLike(*bottom[i*2]);
}

template <typename Dtype>
void SuperCategoryInverseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	for(int i = 0; i < depth_; ++i)
		caffe_copy(top[i]->count(), bottom[i*2]->cpu_data(), top[i]->mutable_cpu_data());

	for(int m = 0; m < M_; ++m) {
		for(int i = 1; i < depth_; ++i) {
			int base_idx = base_index_per_level_[i-1];
			for(int j = 0; j < node_num_per_level_[i-1]; ++j) {
				Tree * node = serialized_tree_[base_idx + j];
				const std::vector<shared_ptr<Tree> > * children = node->GetChildren();
				Dtype value = top[i-1]->data_at(m,node->GetLabel(),0,0);

				for(auto it = children->cbegin(); it != children->cend(); ++it) {
					int offset = top[i]->offset(m,(*it)->GetLabel());
					top[i]->mutable_cpu_data()[offset] += value;

				}
			}
		}
	}
}

template <typename Dtype>
void SuperCategoryInverseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	for(int i = 0; i < depth_; ++i)
		caffe_copy(top[i]->count(), top[i]->cpu_diff(), bottom[i*2]->mutable_cpu_diff());

	for(int m = 0; m < M_; ++m) {
		for(int i = depth_-2; i >= 0; --i) {
			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j) {
				Tree * node = serialized_tree_[base_idx + j];
				const std::vector<shared_ptr<Tree> > * children = node->GetChildren();
				
				int offset = bottom[i*2]->offset(m,node->GetLabel());
				Dtype * diff = bottom[i*2]->mutable_cpu_diff() + offset;
				for(auto it = children->cbegin(); it != children->cend(); ++it) {
					*diff += bottom[(i+1)*2]->diff_at(m,(*it)->GetLabel(),0,0);
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(SuperCategoryInverseLayer);
#endif

INSTANTIATE_CLASS(SuperCategoryInverseLayer);
REGISTER_LAYER_CLASS(SuperCategoryInverse);
}  // namespace caffe
