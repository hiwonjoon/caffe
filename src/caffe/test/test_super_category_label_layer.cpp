#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class SuperCategoryLabelLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SuperCategoryLabelLayerTest()
	  : blob_bottom_label_(new Blob<Dtype>(5, 1, 1, 1))
  {
	//add to vector
    blob_bottom_vec_.push_back(blob_bottom_label_);

	// make the top
	for(int i = 0; i < 3; ++i)
		blob_top_vec_.push_back(new Blob<Dtype>());
	// make the top
	for(int i = 0; i < 5; ++i)
		blob_top_vec_hard_.push_back(new Blob<Dtype>());
  }
  virtual ~SuperCategoryLabelLayerTest() { 
	  delete blob_bottom_label_;
	  for(auto it = blob_top_vec_.begin(); it != blob_top_vec_.end(); ++it)
		  delete *it;
	  for(auto it = blob_top_vec_hard_.begin(); it != blob_top_vec_hard_.end(); ++it)
		  delete *it;
  }
  Blob<Dtype>* const blob_bottom_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_top_vec_hard_;

  void SetSuperCategoryParam(SuperCategoryParameter * sup_param) {
	  blob_bottom_label_->Reshape(5,1,1,1);

	  SuperCategoryParameter::TreeScheme * root = sup_param->mutable_root();
	  SuperCategoryParameter::TreeScheme * child1 = root->add_children();
	  child1->set_label(0);
	  SuperCategoryParameter::TreeScheme * child2 = root->add_children();
	  SuperCategoryParameter::TreeScheme * child2_1 = child2->add_children();
	  child2_1->set_label(1);
	  SuperCategoryParameter::TreeScheme * child2_2 = child2->add_children();
	  child2_2->set_label(2);
	  SuperCategoryParameter::TreeScheme * child2_3 = child2->add_children();
	  SuperCategoryParameter::TreeScheme * child2_3_1 = child2_3->add_children();
	  child2_3_1->set_label(3);
	  SuperCategoryParameter::TreeScheme * child2_3_2 = child2_3->add_children();
	  child2_3_2->set_label(4);

	  //weight_filler
	  sup_param->mutable_weight_filler()->set_type("uniform");
	  sup_param->mutable_weight_filler()->set_min(1);
	  sup_param->mutable_weight_filler()->set_max(2);
  }
  void SetSuperCategoryParam_Hard(SuperCategoryParameter * sup_param) {
	  blob_bottom_label_->Reshape(13,1,1,1);

	  SuperCategoryParameter::TreeScheme * root = sup_param->mutable_root();
	  SuperCategoryParameter::TreeScheme * child1 = root->add_children();
	  SuperCategoryParameter::TreeScheme * child1_1 = child1->add_children();
	  child1_1->set_label(12);
	  SuperCategoryParameter::TreeScheme * child1_2 = child1->add_children();
	  SuperCategoryParameter::TreeScheme * child1_2_1 = child1_2->add_children();
	  child1_2_1->set_label(0);
	  SuperCategoryParameter::TreeScheme * child1_2_2 = child1_2->add_children();
	  child1_2_2->set_label(1);
	  SuperCategoryParameter::TreeScheme * child2 = root->add_children();
	  SuperCategoryParameter::TreeScheme * child2_1 = child2->add_children();
	  child2_1->set_label(5);
	  SuperCategoryParameter::TreeScheme * child2_2 = child2->add_children();
	  child2_2->set_label(3);
	  SuperCategoryParameter::TreeScheme * child2_3 = child2->add_children();
	  SuperCategoryParameter::TreeScheme * child2_3_1 = child2_3->add_children();
	  child2_3_1->set_label(4);
	  SuperCategoryParameter::TreeScheme * child2_3_2 = child2_3->add_children();
	  SuperCategoryParameter::TreeScheme * child2_3_2_1 = child2_3_2->add_children();
	  child2_3_2_1->set_label(2);
	  SuperCategoryParameter::TreeScheme * child2_3_2_2 = child2_3_2->add_children();
	  child2_3_2_2->set_label(6);
	  SuperCategoryParameter::TreeScheme * child2_3_3 = child2_3->add_children();
	  SuperCategoryParameter::TreeScheme * child2_3_3_1 = child2_3_3->add_children();
	  child2_3_3_1->set_label(7);
	  SuperCategoryParameter::TreeScheme * child2_3_3_2 = child2_3_3->add_children();
	  child2_3_3_2->set_label(8);
	  SuperCategoryParameter::TreeScheme * child2_3_3_3 = child2_3_3->add_children();
	  child2_3_3_3->set_label(9);
	  SuperCategoryParameter::TreeScheme * child2_3_3_4 = child2_3_3->add_children();
	  SuperCategoryParameter::TreeScheme * child2_3_3_4_1 = child2_3_3_4->add_children();
	  child2_3_3_4_1->set_label(10);
	  SuperCategoryParameter::TreeScheme * child2_3_3_4_2 = child2_3_3_4->add_children();
	  child2_3_3_4_2->set_label(11);

	  //weight_filler
	  sup_param->mutable_weight_filler()->set_type("uniform");
	  sup_param->mutable_weight_filler()->set_min(1);
	  sup_param->mutable_weight_filler()->set_max(2);
  }
};

TYPED_TEST_CASE(SuperCategoryLabelLayerTest, TestDtypesAndDevices);

TYPED_TEST(SuperCategoryLabelLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  this->SetSuperCategoryParam(layer_param.mutable_super_category_param());

  shared_ptr<SuperCategoryLabelLayer<Dtype> > layer(
      new SuperCategoryLabelLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_vec_[0]->num(),5);
  EXPECT_EQ(this->blob_top_vec_[0]->height(),1);
  EXPECT_EQ(this->blob_top_vec_[0]->width(),1);
  EXPECT_EQ(this->blob_top_vec_[0]->channels(),1);
  EXPECT_EQ(this->blob_top_vec_[1]->num(),5);
  EXPECT_EQ(this->blob_top_vec_[1]->height(),1);
  EXPECT_EQ(this->blob_top_vec_[1]->width(),1);
  EXPECT_EQ(this->blob_top_vec_[1]->channels(),1);
  EXPECT_EQ(this->blob_top_vec_[2]->num(),5);
  EXPECT_EQ(this->blob_top_vec_[2]->height(),1);
  EXPECT_EQ(this->blob_top_vec_[2]->width(),1);
  EXPECT_EQ(this->blob_top_vec_[2]->channels(),1);
}

TYPED_TEST(SuperCategoryLabelLayerTest, TestForwardLabel) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	this->SetSuperCategoryParam(layer_param.mutable_super_category_param());
	shared_ptr<SuperCategoryLabelLayer<Dtype> > layer(
	  new SuperCategoryLabelLayer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

	this->blob_bottom_label_->mutable_cpu_data()[0] = 0;
	this->blob_bottom_label_->mutable_cpu_data()[1] = 1;
	this->blob_bottom_label_->mutable_cpu_data()[2] = 2;
	this->blob_bottom_label_->mutable_cpu_data()[3] = 3;
	this->blob_bottom_label_->mutable_cpu_data()[4] = 4;

	layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	EXPECT_EQ(this->blob_top_vec_[0]->cpu_data()[0], 0);
	EXPECT_EQ(this->blob_top_vec_[1]->cpu_data()[0], 0);
	EXPECT_EQ(this->blob_top_vec_[2]->cpu_data()[0], 0);

	EXPECT_EQ(this->blob_top_vec_[0]->cpu_data()[1], 1);
	EXPECT_EQ(this->blob_top_vec_[1]->cpu_data()[1], 1);
	EXPECT_EQ(this->blob_top_vec_[2]->cpu_data()[1], 1);

	EXPECT_EQ(this->blob_top_vec_[0]->cpu_data()[2], 1);
	EXPECT_EQ(this->blob_top_vec_[1]->cpu_data()[2], 2);
	EXPECT_EQ(this->blob_top_vec_[2]->cpu_data()[2], 2);

	EXPECT_EQ(this->blob_top_vec_[0]->cpu_data()[3], 1);
	EXPECT_EQ(this->blob_top_vec_[1]->cpu_data()[3], 3);
	EXPECT_EQ(this->blob_top_vec_[2]->cpu_data()[3], 3);

	EXPECT_EQ(this->blob_top_vec_[0]->cpu_data()[4], 1);
	EXPECT_EQ(this->blob_top_vec_[1]->cpu_data()[4], 3);
	EXPECT_EQ(this->blob_top_vec_[2]->cpu_data()[4], 4);
}

TYPED_TEST(SuperCategoryLabelLayerTest, TestForwardLabel_Hard) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	this->SetSuperCategoryParam_Hard(layer_param.mutable_super_category_param());
	shared_ptr<SuperCategoryLabelLayer<Dtype> > layer(
	  new SuperCategoryLabelLayer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_hard_);

	for(int i = 0; i < 13; ++i)
		this->blob_bottom_label_->mutable_cpu_data()[i] = i;

	layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_hard_);
	EXPECT_EQ(this->blob_top_vec_hard_[4]->cpu_data()[0], 0);
	EXPECT_EQ(this->blob_top_vec_hard_[3]->cpu_data()[0], 1);
	EXPECT_EQ(this->blob_top_vec_hard_[2]->cpu_data()[0], 1);
	EXPECT_EQ(this->blob_top_vec_hard_[1]->cpu_data()[0], 1);
	EXPECT_EQ(this->blob_top_vec_hard_[0]->cpu_data()[0], 0);

	EXPECT_EQ(this->blob_top_vec_hard_[4]->cpu_data()[1], 1);
	EXPECT_EQ(this->blob_top_vec_hard_[3]->cpu_data()[1], 2);
	EXPECT_EQ(this->blob_top_vec_hard_[2]->cpu_data()[1], 2);
	EXPECT_EQ(this->blob_top_vec_hard_[1]->cpu_data()[1], 1);
	EXPECT_EQ(this->blob_top_vec_hard_[0]->cpu_data()[1], 0);

	EXPECT_EQ(this->blob_top_vec_hard_[4]->cpu_data()[2], 2);
	EXPECT_EQ(this->blob_top_vec_hard_[3]->cpu_data()[2], 6);
	EXPECT_EQ(this->blob_top_vec_hard_[2]->cpu_data()[2], 6);
	EXPECT_EQ(this->blob_top_vec_hard_[1]->cpu_data()[2], 4);
	EXPECT_EQ(this->blob_top_vec_hard_[0]->cpu_data()[2], 1);

	EXPECT_EQ(this->blob_top_vec_hard_[4]->cpu_data()[3], 3);
	EXPECT_EQ(this->blob_top_vec_hard_[3]->cpu_data()[3], 4);
	EXPECT_EQ(this->blob_top_vec_hard_[2]->cpu_data()[3], 4);
	EXPECT_EQ(this->blob_top_vec_hard_[1]->cpu_data()[3], 3);
	EXPECT_EQ(this->blob_top_vec_hard_[0]->cpu_data()[3], 1);

	EXPECT_EQ(this->blob_top_vec_hard_[4]->cpu_data()[4], 4);
	EXPECT_EQ(this->blob_top_vec_hard_[3]->cpu_data()[4], 5);
	EXPECT_EQ(this->blob_top_vec_hard_[2]->cpu_data()[4], 5);
	EXPECT_EQ(this->blob_top_vec_hard_[1]->cpu_data()[4], 4);
	EXPECT_EQ(this->blob_top_vec_hard_[0]->cpu_data()[4], 1);

	EXPECT_EQ(this->blob_top_vec_hard_[4]->cpu_data()[5], 5);
	EXPECT_EQ(this->blob_top_vec_hard_[3]->cpu_data()[5], 3);
	EXPECT_EQ(this->blob_top_vec_hard_[2]->cpu_data()[5], 3);
	EXPECT_EQ(this->blob_top_vec_hard_[1]->cpu_data()[5], 2);
	EXPECT_EQ(this->blob_top_vec_hard_[0]->cpu_data()[5], 1);

	EXPECT_EQ(this->blob_top_vec_hard_[4]->cpu_data()[6], 6);
	EXPECT_EQ(this->blob_top_vec_hard_[3]->cpu_data()[6], 7);
	EXPECT_EQ(this->blob_top_vec_hard_[2]->cpu_data()[6], 6);
	EXPECT_EQ(this->blob_top_vec_hard_[1]->cpu_data()[6], 4);
	EXPECT_EQ(this->blob_top_vec_hard_[0]->cpu_data()[6], 1);

	EXPECT_EQ(this->blob_top_vec_hard_[4]->cpu_data()[7], 7);
	EXPECT_EQ(this->blob_top_vec_hard_[3]->cpu_data()[7], 8);
	EXPECT_EQ(this->blob_top_vec_hard_[2]->cpu_data()[7], 7);
	EXPECT_EQ(this->blob_top_vec_hard_[1]->cpu_data()[7], 4);
	EXPECT_EQ(this->blob_top_vec_hard_[0]->cpu_data()[7], 1);

	EXPECT_EQ(this->blob_top_vec_hard_[4]->cpu_data()[8], 8);
	EXPECT_EQ(this->blob_top_vec_hard_[3]->cpu_data()[8], 9);
	EXPECT_EQ(this->blob_top_vec_hard_[2]->cpu_data()[8], 7);
	EXPECT_EQ(this->blob_top_vec_hard_[1]->cpu_data()[8], 4);
	EXPECT_EQ(this->blob_top_vec_hard_[0]->cpu_data()[8], 1);

	EXPECT_EQ(this->blob_top_vec_hard_[4]->cpu_data()[9], 9);
	EXPECT_EQ(this->blob_top_vec_hard_[3]->cpu_data()[9], 10);
	EXPECT_EQ(this->blob_top_vec_hard_[2]->cpu_data()[9], 7);
	EXPECT_EQ(this->blob_top_vec_hard_[1]->cpu_data()[9], 4);
	EXPECT_EQ(this->blob_top_vec_hard_[0]->cpu_data()[9], 1);

	EXPECT_EQ(this->blob_top_vec_hard_[4]->cpu_data()[10], 10);
	EXPECT_EQ(this->blob_top_vec_hard_[3]->cpu_data()[10], 11);
	EXPECT_EQ(this->blob_top_vec_hard_[2]->cpu_data()[10], 7);
	EXPECT_EQ(this->blob_top_vec_hard_[1]->cpu_data()[10], 4);
	EXPECT_EQ(this->blob_top_vec_hard_[0]->cpu_data()[10], 1);

	EXPECT_EQ(this->blob_top_vec_hard_[4]->cpu_data()[11], 11);
	EXPECT_EQ(this->blob_top_vec_hard_[3]->cpu_data()[11], 11);
	EXPECT_EQ(this->blob_top_vec_hard_[2]->cpu_data()[11], 7);
	EXPECT_EQ(this->blob_top_vec_hard_[1]->cpu_data()[11], 4);
	EXPECT_EQ(this->blob_top_vec_hard_[0]->cpu_data()[11], 1);

	EXPECT_EQ(this->blob_top_vec_hard_[4]->cpu_data()[12], 12); //for label 12
	EXPECT_EQ(this->blob_top_vec_hard_[3]->cpu_data()[12], 0);
	EXPECT_EQ(this->blob_top_vec_hard_[2]->cpu_data()[12], 0);
	EXPECT_EQ(this->blob_top_vec_hard_[1]->cpu_data()[12], 0);
	EXPECT_EQ(this->blob_top_vec_hard_[0]->cpu_data()[12], 0);
}

}  // namespace caffe
