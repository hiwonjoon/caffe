#include <queue>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/io.hpp"

namespace caffe {
//Tree Implementation
int Tree::Depth() const {
	int max_depth = 0;
	for(int i = 0; i < this->children.size(); i++) {
		int depth = this->children[i]->Depth(); if( max_depth < depth ) max_depth = depth; }
	return max_depth + 1;
}
void Tree::MakeBalance(int remain) {
	if( remain == 0 ) return;
	if( children.size() == 0 ) {
		Tree * root = this;
		int label = root->label;
		for(int i = 0; i < remain; ++i ) {
			root->InsertChild(shared_ptr<Tree>(new Tree()));
			root->SetLabel(-1);
			root = root->children[0].get();
		}
		root->SetLabel(label);
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
void Tree::GetNodeNumPerLevelAndGiveLabel(std::vector<int>& node_num, std::vector<int>& base_index,Tree * root, std::vector<Tree *>& serialized_tree, std::vector<int>& label_to_index) {
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

		if( i < depth-2 ){ //label for last layer is already made
			for(int j = base_index[i]; j < base_index[i]+node_num[i]; ++j)
				serialized_tree[j]->label = j - base_index[i];
		}
		else {
			label_to_index.resize(node_num[i]);
			for(int index = 0; index < node_num[i]; ++index) {
				int label = serialized_tree[index+base_index[i]]->GetLabel();
				label_to_index[label] = index;
			}
		}
	}
}
void Tree::MakeTree(Tree * node, const SuperCategoryParameter::TreeScheme * node_param){
	if( node_param->children_size() == 0 ){
		CHECK_NE(node_param->label(),-1);
		node->SetLabel(node_param->label());
	}
	else {
		CHECK_EQ(node->label,-1);
		for(int i = 0; i < node_param->children_size(); ++i) {
			shared_ptr<Tree> child(new Tree());
			node->InsertChild(child);
			MakeTree(child.get(), &node_param->children(i));
		}
	}
}

}
