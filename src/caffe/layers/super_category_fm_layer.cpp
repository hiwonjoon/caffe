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

	std::vector<int> shape;
	shape.push_back(M_);
	shape.push_back(1);
	shape.push_back(H_);
	shape.push_back(W_);

	temp_blobs.resize(depth_);
	for(int i = 0; i < depth_; ++i) {
		temp_blobs[i].resize(node_num_per_level_[i]);
		for(int j = 0; j < node_num_per_level_[i]; ++j) {
			temp_blobs[i][j].reset(new Blob<Dtype>(shape));
		}
	}
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
	for(int i = 0; i < depth_; ++i) {
		for(int j = 0; j < node_num_per_level_[i]; ++j) {
			if( i == depth_-1 ) {
				for(int m = 0; m < M_; ++m) {
					caffe_copy(H_*W_, &bottom[0]->cpu_data()[bottom[0]->offset(m,j)],&temp_blobs[i][j]->mutable_cpu_data()[temp_blobs[i][j]->offset(m)]);
				}
			}
			else
				caffe_set(temp_blobs[i][j]->count(), (Dtype)0., temp_blobs[i][j]->mutable_cpu_data());
		}
	}

	for( int i = depth_-2; i >= 0; --i ) {
		int base_idx = base_index_per_level_[i];
		for(int j = 0; j < node_num_per_level_[i]; ++j) {
			Tree * node = serialized_tree_[base_idx + j];
			const std::vector<shared_ptr<Tree> >* children = node->GetChildren();

			shared_ptr<Blob<Dtype>> tops = temp_blobs[i][node->GetLabel()];
			Dtype * top_data = tops->mutable_cpu_data();

			for(auto it = children->cbegin(); it != children->cend(); ++it) {
				shared_ptr<Blob<Dtype>> bottoms = temp_blobs[i+1][(*it)->GetLabel()];
				const Dtype * bottom_data = bottoms->cpu_data();
				caffe_axpy(M_*H_*W_,(Dtype)(1.),bottom_data,top_data);
			}
			caffe_scal(M_*H_*W_,(Dtype)(1./children->size()),top_data);
		}
	}

	for(int m = 0; m < M_; ++m) {
		for(int i = 0; i < depth_; ++i) {
			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j) {
				Tree * node = serialized_tree_[base_idx + j];
				caffe_copy(H_*W_,&temp_blobs[i][node->GetLabel()]->cpu_data()[temp_blobs[i][node->GetLabel()]->offset(m)],&top[i]->mutable_cpu_data()[top[i]->offset(m,node->GetLabel())]);
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
		for(int i = 0; i < depth_; ++i) {
			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j) {
				Tree * node = serialized_tree_[base_idx + j];
				caffe_copy(H_*W_,&top[i]->cpu_diff()[top[i]->offset(m,node->GetLabel())],&temp_blobs[i][node->GetLabel()]->mutable_cpu_diff()[temp_blobs[i][node->GetLabel()]->offset(m)]);
			}
		}
	}

	for( int i = 0; i < depth_-1; ++i ) {
		int base_idx = base_index_per_level_[i];
		for(int j = 0; j < node_num_per_level_[i]; ++j) {
			Tree * node = serialized_tree_[base_idx + j];
			const std::vector<shared_ptr<Tree> >* children = node->GetChildren();

			shared_ptr<Blob<Dtype>> tops = temp_blobs[i][node->GetLabel()];
			const Dtype * top_diff = tops->cpu_diff();
			for(auto it = children->cbegin(); it != children->cend(); ++it) {
				shared_ptr<Blob<Dtype>> bottoms = temp_blobs[i+1][(*it)->GetLabel()];
				Dtype * bottom_diff = bottoms->mutable_cpu_diff();

				caffe_axpy(M_*H_*W_,(Dtype)(1./children->size()),top_diff,bottom_diff);	
			}

		}
	}

	for(int m = 0; m < M_; ++m) {
		for(int i = 0; i < depth_; ++i) {
			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j) {
				Tree * node = serialized_tree_[base_idx + j];
				caffe_copy(H_*W_,&temp_blobs[i][node->GetLabel()]->cpu_diff()[temp_blobs[i][node->GetLabel()]->offset(m)],&top[i]->mutable_cpu_diff()[top[i]->offset(m,node->GetLabel())]);
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

