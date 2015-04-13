#include <iostream>
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
void RegularizeLayer<Dtype>::analyze_tree() {
	const RegularizeParameter regu_param = this->layer_param_.regularize_param();
	const RegularizeParameter::TreeScheme * root = &regu_param.root();
	internal_node_ = count_internal_node(root);
	node_ = count_node(root);
}
template <typename Dtype>
int RegularizeLayer<Dtype>::count_internal_node(const RegularizeParameter::TreeScheme * root) {
	int temp_cnt = root->children_size() == 0 ? 0 : 1;
	if( root->children_size() != 0 )
		internal_nodes_.push_back(root->node_num());

	for(int i = 0; i < root->children_size(); i++)
		temp_cnt += count_internal_node(&root->children(i));
	return temp_cnt;
}
template <typename Dtype>
int RegularizeLayer<Dtype>::count_node(const RegularizeParameter::TreeScheme * root) {
	int temp_cnt = 1;
	for(int i = 0; i < root->children_size(); i++)
		temp_cnt += count_node(&root->children(i));
	return temp_cnt;
}
template <typename Dtype>
void RegularizeLayer<Dtype>::make_tree_map() {
	const RegularizeParameter regu_param = this->layer_param_.regularize_param();
	for(int i = 1; i <= node_; i++)
	{
		Dtype * map = tree_map_.mutable_cpu_data() + (i-1)*N_;
		const RegularizeParameter::TreeScheme * root = &regu_param.root();
		mapping(map,root,i,false); 
	}


	//const Dtype * tree_map_data = tree_map_.cpu_data();
	//for(int i = 1; i <= node_; i++)
	//{
	//	for(int j = 1; j <= N_; j++) {
	//		std::cout << (tree_map_data[N_*(i-1)+(j-1)]) << " ";
	//	}
	//	std::cout << std::endl;
	//}
}
template <typename Dtype>
void RegularizeLayer<Dtype>::mapping(Dtype * map, const RegularizeParameter::TreeScheme * root, int current_node, bool flag) {
	if( flag == false && current_node == root->node_num() )
		flag = true;

	if( root->children_size() == 0 )
		map[root->output_index()-1] = flag ? 1. : 0.;
	for(int i = 0; i < root->children_size(); i++)
		mapping(map,&root->children(i),current_node,flag);
}

template <typename Dtype>
void RegularizeLayer<Dtype>::make_g_agg()
{
	const RegularizeParameter regu_param = this->layer_param_.regularize_param();
	Dtype * g_agg_data = g_agg_.mutable_cpu_data();
	const RegularizeParameter::TreeScheme * root = &regu_param.root();

	for(int i = 1; i <= node_; i++)
	{
		(g_agg_data)[i-1] = make_g_agg_rec(root,i);
	}
}

template <typename Dtype>
Dtype RegularizeLayer<Dtype>::make_g_agg_rec(const RegularizeParameter::TreeScheme * root, int current_node) {
	if( root->children_size() == 0 && current_node != root->node_num() )
		return 0;

	if( current_node == root->node_num() )
	{
		if( root->children_size() == 0 )
			return 1;
		else
			return (this->blobs_[0]->cpu_data())[current_node-1];
	}

	for( int i = 0; i < root->children_size(); i++)
	{
		Dtype agg = make_g_agg_rec(&root->children(i),current_node);
		if( agg != 0 )
			return agg * (1 - (this->blobs_[0]->cpu_data())[root->node_num()-1]);
	}
	return 0;
}

template <typename Dtype>
void RegularizeLayer<Dtype>::make_diff_g(Blob<Dtype>& diff_g)
{
	const RegularizeParameter regu_param = this->layer_param_.regularize_param();
	Dtype* diff_g_data = diff_g.mutable_cpu_data();
	const RegularizeParameter::TreeScheme * root = &regu_param.root();

	for(int i = 0; i < internal_node_; ++i) {
		for (int j = 0; j < node_; ++j) {
			diff_g_data[ i * node_ + j] = make_diff_g_rec(root, internal_nodes_[i], j+1, false);
		}
	}
}

template <typename Dtype>
Dtype RegularizeLayer<Dtype>::make_diff_g_rec(const RegularizeParameter::TreeScheme * root, int diff_node, int current_node, bool flag) {
	if( root->node_num() == diff_node )
	{
		if( root->children_size() == 0 )
			throw "WTF!";	//TODO : Use assert.
		flag = true;

		if( diff_node == current_node )
			return 1;

		for( int i = 0; i < root->children_size(); i++ )
		{
			Dtype result = make_diff_g_rec(&root->children(i),diff_node,current_node,flag);
			if( result != 0. )
				return -1 * result;
		}
		return 0;
	}
	else
	{
		if( root->children_size() == 0 && current_node != root->node_num())
			return 0;

		if( current_node == root->node_num() && flag )
		{
			if( root->children_size() == 0 )
				return 1;
			else
				return this->blobs_[0]->cpu_data()[root->node_num()-1];
		}
		
		for( int i = 0; i < root->children_size(); i++)
		{
			Dtype result = make_diff_g_rec(&root->children(i),diff_node,current_node,flag);
			if( result != 0. )
			{
				return result * (1 - (this->blobs_[0]->cpu_data())[root->node_num()-1]);
			}
		}
		return 0.;
	}
}

template <typename Dtype>
void RegularizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	//TODO : node_ == output_node 개수
	analyze_tree();

	//bottom_ = bottom;
	N_ = bottom[0]->shape(0);
	K_ = bottom[0]->shape(1);

	this->blobs_.resize(1);
	this->blobs_[0].reset(new Blob<Dtype>(node_,1,1,1));
	shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(this->layer_param_.regularize_param().weight_filler()));
	weight_filler->Fill(this->blobs_[0].get());

//	const Dtype * g_data = this->blobs_[0]->cpu_data();
//	for(int i = 0; i < node_; i++){
//		std::cout << g_data[i] << " ";
//	}
//	std::cout << std::endl;

	//param propagate down??
	tree_map_.Reshape(node_,N_,1,1);
	gv_map_.Reshape(node_,K_,1,1);
	g_agg_.Reshape(node_,1,1,1);

	make_tree_map();

	this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void RegularizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	N_ = bottom[0]->shape(0);
	const int new_K = bottom[0]->shape(1);
	CHECK_EQ(K_,new_K) << "input size is not compatible";
	K_ = new_K;

	top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void RegularizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Blob<Dtype> temp;
	temp.CopyFrom(*bottom[0],false,true);

	const int count = bottom[0]->count();
	Dtype * temp_data = temp.mutable_cpu_data();
	caffe_powx(count,temp_data,(Dtype)2,temp_data);

	//for(int i = 0 ; i < bottom[0]->shape(0); ++i) {
	//	for (int j = 0; j < bottom[0]->shape(1); ++j) {
	//		std::cout << temp_data[i*bottom[0]->shape(1)+j] << " ";
	//	}
	//	std::cout << std::endl;
	//}

	const Dtype * tree_map_data = tree_map_.cpu_data();
	Dtype* gv_map_data = gv_map_.mutable_cpu_data();		

	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, node_, K_, N_, (Dtype)1., tree_map_data, temp.cpu_data(), (Dtype)0., gv_map_data);

	//for(int i = 0; i < node_;  ++i) {
	//	for(int j = 0; j < K_; ++j) {
	//		std::cout << gv_map_data[i*K_+j] << " ";
	//	}
	//	std::cout << std::endl;
	//}

	caffe_powx( gv_map_.count(), gv_map_.cpu_data(), (Dtype)0.5, gv_map_.mutable_cpu_data());
	
	//for(int i = 0; i < node_;  ++i) {
	//	for(int j = 0; j < K_; ++j) {
	//		std::cout << gv_map_data[i*K_+j] << " ";
	//	}
	//	std::cout << std::endl;
	//}

	make_g_agg();

	const Dtype * g_agg_data = g_agg_.cpu_data();

	//for(int i = 0; i < node_; i++)
	//{
	//	std::cout << g_agg_data[i] << " "; 
	//}
	//std::cout << std::endl;

	Blob<Dtype> temp2(1, K_, 1, 1);
	Dtype* temp2_data = temp2.mutable_cpu_data();
	caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 1, K_, node_, (Dtype)1., g_agg_data, gv_map_.cpu_data(), (Dtype)0., temp2_data);

	
	//for(int i = 0; i < K_; i++) {
	//	std::cout << temp2_data[i] << " ";
	//}
	//std::cout << std::endl;

	Dtype sum = 0.;
	for(int i = 0; i < K_; i++)
	{
		sum += temp2_data[i];	
	}
	(top[0]->mutable_cpu_data())[0] = sum;
}

template <typename Dtype>
void RegularizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype * top_diff = top[0]->cpu_diff();

	//gradient with respect to g_ coefficient
	Blob<Dtype> temp2(node_,K_,1,1);
	Blob<Dtype> diff_g(internal_node_, node_, 1, 1);
	make_diff_g(diff_g);

	//const Dtype * diff_g_data = diff_g.cpu_data();
	//for(int i = 0; i < internal_node_; ++i) {
	//	for(int j = 0; j < node_; ++j) {
	//		std::cout << diff_g_data[i* node_ + j] << " " ;
	//	}
	//	std::cout << std::endl;
	//}
	
	caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,internal_node_,K_,node_, *top_diff, diff_g.cpu_data(),gv_map_.cpu_data(), (Dtype)0., temp2.mutable_cpu_data()); 

	Dtype * g_diff = this->blobs_[0]->mutable_cpu_diff();
	const Dtype * temp2_data = temp2.cpu_data();
	for( int i = 0; i < internal_node_; i++ ){
		Dtype sum = 0.;
		for( int j = 0; j < K_; j++) {
			sum += temp2_data[i*K_+j];
		}
		g_diff[internal_nodes_[i]-1] = sum;	
	}

	//gradient with bottom data
	if( propagate_down[0] )
	{
		int count = gv_map_.count();
		Dtype* gv_map_data = gv_map_.mutable_cpu_data();
		caffe_powx(count, gv_map_data, (Dtype)-1, gv_map_data);

		Blob<Dtype> temp;
		temp.CopyFrom(tree_map_,false,true);
		Dtype * temp_data = temp.mutable_cpu_data();
		for(int i = 0; i < node_; i++)
		{
			for(int j = 0; j < N_; j++)
			{
				temp_data[i*N_+j] *= (g_agg_.cpu_data())[i];
			}
		}

		//for(int i = 0; i < node_; ++i)
		//{
		//	for(int j = 0; j < N_; ++j)
		//		std::cout << temp_data[i*N_+j] << " ";
		//	std::cout << std::endl;
		//}
		
		//TODO : use weight sharing?, (Dtype)1.0 or 0.0? 
		std::vector<int> shape;
		shape.push_back(N_);
		shape.push_back(K_);
		temp2.Reshape(shape);
		caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, node_, *top_diff, temp_data, gv_map_data, (Dtype)0., temp2.mutable_cpu_diff());
		caffe_mul(temp2.count(), bottom[0]->cpu_data(), temp2.cpu_diff(), temp2.mutable_cpu_diff());
		caffe_add(temp2.count(), bottom[0]->cpu_diff(), temp2.cpu_diff(), bottom[0]->mutable_cpu_diff());
	}

}


#ifdef CPU_ONLY
//STUB_GPU(RegularizeLayer);
#endif

INSTANTIATE_CLASS(RegularizeLayer);
REGISTER_LAYER_CLASS(Regularize);
}
