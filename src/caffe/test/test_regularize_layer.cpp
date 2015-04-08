#include <cstring>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class RegularizeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

	protected:
  RegularizeLayerTest()
	  : blob_bottom_(new Blob<Dtype>(2, 4096, 1, 1)),
		blob_top_(new Blob<Dtype>(1,1,1,1))
  {
	  FillerParameter filler_param;
	  //filler_param.set_value(-1);
	  //ConstantFiller<Dtype> filler(filler_param);
	  filler_param.set_min(1);
	  filler_param.set_max(2);
	  UniformFiller<Dtype> filler(filler_param);
	  //GaussianFiller<Dtype> filler(filler_param);
	  
	  filler.Fill(this->blob_bottom_);
	  blob_bottom_vec_.push_back(blob_bottom_);
	  blob_top_vec_.push_back(blob_top_);
  }

  virtual ~RegularizeLayerTest() {
	  delete blob_bottom_;
	  delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RegularizeLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(RegularizeLayerTest, DoubleCPU);

TYPED_TEST(RegularizeLayerTest, TestForward) {
	typedef typename TypeParam::Dtype Dtype;

	LayerParameter layer_param;
	RegularizeParameter* regu_param = layer_param.mutable_regularize_param();
	
	RegularizeParameter::TreeScheme * root = regu_param->mutable_root();
	root->set_node_num(1);
	RegularizeParameter::TreeScheme * child1 = root->add_children();
	child1->set_node_num(2);
	child1->set_output_index(1);
	RegularizeParameter::TreeScheme * child2 = root->add_children();
	child2->set_node_num(3);
	child2->set_output_index(2);

	//regu_param->mutable_weight_filler()->set_type("constant");
	//regu_param->mutable_weight_filler()->set_value(0.5);
	regu_param->mutable_weight_filler()->set_type("uniform");
	regu_param->mutable_weight_filler()->set_min(0);
	regu_param->mutable_weight_filler()->set_max(1);

	shared_ptr<Layer<Dtype> > layer(
		new RegularizeLayer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

	const Dtype * data = this->blob_top_->cpu_data();
	std::cout << *data << std::endl;
}

TYPED_TEST(RegularizeLayerTest, TestBackward) {
	typedef typename TypeParam::Dtype Dtype;

	LayerParameter layer_param;
	RegularizeParameter* regu_param = layer_param.mutable_regularize_param();
	
	RegularizeParameter::TreeScheme * root = regu_param->mutable_root();
	root->set_node_num(1);
	RegularizeParameter::TreeScheme * child1 = root->add_children();
	child1->set_node_num(2);
	child1->set_output_index(1);
	RegularizeParameter::TreeScheme * child2 = root->add_children();
	child2->set_node_num(3);
	child2->set_output_index(2);

	//regu_param->mutable_weight_filler()->set_type("constant");
	//regu_param->mutable_weight_filler()->set_value(0.5);
	regu_param->mutable_weight_filler()->set_type("uniform");
	regu_param->mutable_weight_filler()->set_min(0);
	regu_param->mutable_weight_filler()->set_max(1);

	shared_ptr<Layer<Dtype> > layer(
		new RegularizeLayer<Dtype>(layer_param));
	//layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	//layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

	const Dtype * data = this->blob_top_->cpu_data();
	//std::cout << *data << std::endl;

	(this->blob_top_->mutable_cpu_diff())[0] = *data;

	//vector<bool> propagate_down;
	//propagate_down.resize(1);
	//propagate_down[0] = true;
	//layer->Backward( this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);

    GradientChecker<Dtype> checker(1e-2, 1e-1);
    checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_,this->blob_top_vec_);
}

}
