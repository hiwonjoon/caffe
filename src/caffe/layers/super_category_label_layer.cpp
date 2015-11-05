#include <vector>
#include <limits>
#include <queue>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {
template <typename Dtype>
void SuperCategoryLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	SuperCategoryParameter * super_param = this->layer_param_.mutable_super_category_param();
	if( super_param->file_name().empty() == false ) {
		ReadProtoFromTextFileOrDie(super_param->file_name().c_str(), super_param->mutable_root());
	}

	N_ = bottom[0]->count(0,1);

	Tree::MakeTree(&root_, &super_param->root());
	depth_ = root_.Depth();
	root_.MakeBalance(depth_-1);
	Tree::GiveIndex(&root_, serialized_tree_);
	Tree::GetNodeNumPerLevelAndGiveLabel(node_num_per_level_, base_index_per_level_, &this->root_,serialized_tree_,label_to_index_);
}

template <typename Dtype>
void SuperCategoryLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	CHECK_EQ(top.size(), depth_-1);

	int i = 0;
	for( i = 0; i < depth_-1; ++i) {
		std::vector<int> shape;
		shape.push_back(N_);
		top[i]->Reshape(shape); // Top for label
	}
	CHECK_EQ(bottom[0]->count(), N_);
}

template <typename Dtype>
void SuperCategoryLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	//For Label
	for(int n = 0; n < N_; ++n) {
		int idx = label_to_index_[static_cast<int>(bottom[0]->cpu_data()[n])] + *(base_index_per_level_.rbegin());
		const Tree * node = serialized_tree_[idx];
		for(int i = depth_-2; i >= 0; --i) {
			top[i]->mutable_cpu_data()[n] = node->GetLabel();
			node = node->GetParent();
		}
		CHECK_EQ(top[depth_-2]->cpu_data()[n],bottom[0]->cpu_data()[n]);
	}
}

#ifdef CPU_ONLY
STUB_GPU(SuperCategoryLabelLayer);
#endif

INSTANTIATE_CLASS(SuperCategoryLabelLayer);
REGISTER_LAYER_CLASS(SuperCategoryLabel);

}  // namespace caffe
