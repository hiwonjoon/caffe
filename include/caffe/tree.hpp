#ifndef CAFFE_TREE_HPP_
#define CAFFE_TREE_HPP_

#include <vector>
#include <string>
#include <utility>

#include "caffe/proto/caffe.pb.h"

namespace caffe {
//Tree Class for SuperCategory Layer
class Tree {
public :
  Tree() : label(-1) {}
  ~Tree() {}

  int GetLabel() const { return label; }
  int GetIndex() const { return index; }
  Tree * InsertChild(shared_ptr<Tree> child) {
	children.push_back(child);
	child->parent = this;
	return this;
  }
  void SetLabel(int label_) { this->label = label_; }
  const Tree * GetParent() const { return parent; }
  const std::vector<shared_ptr<Tree> > * GetChildren() const {
	return &children;
  }

  int Depth() const;
  void MakeBalance(int remain);

  //Tree helper
  static void GiveIndex(Tree * root, std::vector<Tree *>& serialized_tree);
  static void GetNodeNumPerLevelAndGiveLabel(std::vector<int>& node_num, std::vector<int>& base_index,Tree * root, std::vector<Tree *>& serialized_tree, std::vector<int>& label_to_index);
  static void MakeTree(Tree * node, const SuperCategoryParameter::TreeScheme * node_param);

private :
  int label;
  int index;

  Tree * parent;
  std::vector<shared_ptr<Tree> > children;
};

} //namespace caffe

#endif // CAFFE_TREE_HPP_
