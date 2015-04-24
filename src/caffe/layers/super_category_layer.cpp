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
//Tree Implementation
int Tree::Depth() const {
	int max_depth = 0;
	for(int i = 0; i < this->children.size(); i++) {
	  int depth = this->children[i]->Depth();
	  if( max_depth < depth ) max_depth = depth;
	}
	return max_depth + 1;
}
void Tree::MakeBalance(int remain) {
	if( remain == 0 ) return;
	if( children.size() == 0 ) {
	  Tree * root = this;
	  for(int i = 0; i < remain; ++i ) {
		  root->InsertChild(shared_ptr<Tree>(new Tree()));
		  root = root->children[0].get();
	  }
	}
	else {
	  for(int i = 0; i < children.size(); ++i)
		  children[i]->MakeBalance(remain-1);
	}
}
//Tree helper
void Tree::GiveIndex(Tree * root, std::vector<Tree *>& serialized_tree) {
	int cnt = 0;
	std::queue<Tree *> queue;
	queue.push(root);
	while( queue.size() != 0 ) {
	  Tree * node = queue.front();
	  node->index = cnt++;
	  serialized_tree.push_back(node);
	  for(int i = 0; i < node->children.size(); ++i)
		  queue.push(node->children[i].get());
	  queue.pop();
	}
}
void Tree::GetNodeNumPerLevel(std::vector<int>& node_num, std::vector<int>& base_index,Tree * root) { 
	Tree * right_root = root;
	int depth = root->Depth();
	node_num.resize(depth-1);
	base_index.resize(depth-1);
	for(int i = 0; i < depth-1; ++i)
	{
	  node_num[i] = right_root->children[right_root->children.size()-1]->GetIndex() - root->children[0]->GetIndex() + 1;
	  base_index[i] = root->children[0]->index;
	  root = root->children[0].get();
	  right_root = right_root->children[right_root->children.size()-1].get();
	}
}
void Tree::MakeTree(Tree * node, const SuperCategoryParameter::TreeScheme * node_param){
	for(int i = 0; i < node_param->children_size(); ++i) {
		shared_ptr<Tree> child(new Tree());
		node->InsertChild(child);
		MakeTree(child.get(), &node_param->children(i));
	}
}

//Layer Implementation
template <typename Dtype>
void SuperCategoryLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const SuperCategoryParameter super_param = this->layer_param_.super_category_param();

	Tree::MakeTree(&root_, &super_param.root());
	root_.MakeBalance(root_.Depth()-1);
	Tree::GiveIndex(&root_, serialized_tree_);
	Tree::GetNodeNumPerLevel(node_num_per_level_, base_index_per_level_, &this->root_);

	N_ = bottom[0]->count(0,1);
	CHECK_EQ(*node_num_per_level_.rbegin(), bottom[0]->count(1));

	this->temp_.Reshape(N_,1,1,1);
}

template <typename Dtype>
void SuperCategoryLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const SuperCategoryParameter super_param = this->layer_param_.super_category_param();

	N_ = bottom[0]->count(0,1);

	Tree::MakeTree(&root_, &super_param.root());
	root_.MakeBalance(root_.Depth()-1);
	Tree::GiveIndex(&root_, serialized_tree_);
	Tree::GetNodeNumPerLevel(node_num_per_level_, base_index_per_level_, &this->root_);
}

template <typename Dtype>
void SuperCategoryLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	CHECK_EQ(top.size(), node_num_per_level_.size());

	mark_.resize(top.size());
	for( int i = 0; i < node_num_per_level_.size(); ++i) {
		top[i]->Reshape(N_,node_num_per_level_[i],1,1); // Top for output data
		mark_[i].reset(new Blob<Dtype> (N_,node_num_per_level_[i],1,1));// Marking for Maxpoolling backprop
	}
}

template <typename Dtype>
void SuperCategoryLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	CHECK_EQ(top.size(), node_num_per_level_.size());

	int i = 0;
	for( i = 0; i < node_num_per_level_.size(); ++i) {
		top[i]->Reshape(N_,1,1,1); // Top for label
	}
	CHECK_EQ(bottom[0]->count(), N_);
}

template <typename Dtype>
void SuperCategoryLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	//For Data
	for(int n = 0; n < N_; ++n) {
		for(int i = node_num_per_level_.size() - 1; i >= 0; --i)
		{
			int node_cnt;
			if( i == node_num_per_level_.size()-1)
				node_cnt = node_num_per_level_[i];
			else
				node_cnt = node_num_per_level_[i+1];

			Blob<Dtype> * bottoms;
			if( i == node_num_per_level_.size() - 1 )
				bottoms = bottom[0];
			else
				bottoms  = top[i+1];

			Dtype * top_data = &top[i]->mutable_cpu_data()[node_num_per_level_[i]*n];
			Dtype * mark_data = &mark_[i]->mutable_cpu_data()[node_num_per_level_[i]*n];
			const Dtype * bottom_data = &bottoms->cpu_data()[node_cnt*n]; //is equal.

			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j ) {
				Tree * node = serialized_tree_[base_idx + j];
				const std::vector<shared_ptr<Tree> > * children = node->GetChildren();
				if( children->size() == 0 )
				{
					CHECK_EQ(i, node_num_per_level_.size() - 1);
					//caffe_mul<Dtype>(N_,&blob_data[N_*j], &bottom_data[N_*j], &top_data[N_*j]);
					top_data[j] = bottom_data[j];
				}
				else{
					top_data[j] = std::numeric_limits<Dtype>::min();
					for(auto it = children->cbegin(); it != children->cend(); ++it) {
						int idx = (*it)->GetIndex() - base_index_per_level_[i+1];
						//caffe_mul<Dtype>(N_,&blob_data[idx*N_],&bottom_data[idx*N_],temp_.mutable_cpu_data());
						//caffe_add<Dtype>(N_,temp_.cpu_data(),&top_data[j*N_],&top_data[j*N_]);
						if( top_data[j] < bottom_data[idx] )
						{
							top_data[j] = bottom_data[idx];
							mark_data[j] = static_cast<Dtype>(idx);
						}
					}
				}
			}
		}
	}
}

template <typename Dtype>
void SuperCategoryLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	//For Label
	for(int n = 0; n < N_; ++n) {
		int node_idx = static_cast<int>(bottom[0]->cpu_data()[n]) + *(base_index_per_level_.rbegin());
		for(int i = node_num_per_level_.size()-1; i >= 0; --i) {
			top[i]->mutable_cpu_data()[n] = node_idx - base_index_per_level_[i];
			node_idx = serialized_tree_[node_idx]->GetParent()->GetIndex();
		}
	}
}

template <typename Dtype>
void SuperCategoryLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());

	for(int n = 0; n < N_; ++n) {
		for(int i = 0; i < node_num_per_level_.size(); ++i) {

			int node_cnt;
			if( i == node_num_per_level_.size()-1)
				node_cnt = node_num_per_level_[i];
			else
				node_cnt = node_num_per_level_[i+1];

			const Dtype * top_diff = &top[i]->cpu_diff()[n*node_num_per_level_[i]];
			const Dtype * mark_data = &mark_[i]->cpu_data()[n*node_num_per_level_[i]];
			Dtype * bottom_diff;
			if( i + 1 == node_num_per_level_.size() ){
				bottom_diff = &bottom[0]->mutable_cpu_diff()[n*node_cnt];
			}
			else {
				bottom_diff = &top[i+1]->mutable_cpu_diff()[n*node_cnt];
			}

			int base_idx = base_index_per_level_[i];
			for(int j = 0; j < node_num_per_level_[i]; ++j) {
				Tree * node = serialized_tree_[base_idx + j];
				const std::vector<shared_ptr<Tree> > * children = node->GetChildren();
				if( propagate_down[0] && children->size() == 0 ) { //this layer is connected with bottom layer
					//caffe_mul<Dtype>(N_,&top_diff[j*N_],&bottom_data[j*N_],&blob_diff[j*N_]);
					//caffe_mul<Dtype>(N_,&top_diff[j*N_],&blob_data[j*N_],&bottom_diff[j*N_]);
					bottom_diff[j] = top_diff[j];
				}
				else {
					int idx = static_cast<int>(mark_data[j]);
					bottom_diff[idx] += top_diff[j];
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(SuperCategoryLayer);
STUB_GPU(SuperCategoryLabelLayer);
#endif

INSTANTIATE_CLASS(SuperCategoryLayer);
REGISTER_LAYER_CLASS(SuperCategory);

INSTANTIATE_CLASS(SuperCategoryLabelLayer);
REGISTER_LAYER_CLASS(SuperCategoryLabel);
}  // namespace caffe
