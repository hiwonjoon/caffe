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
	*(blob_bottom_label_->mutable_cpu_data()) = 1;
	//add to vector
    blob_bottom_vec_.push_back(blob_bottom_label_);

	// make the top
	for(int i = 0; i < 3; ++i)
		blob_top_vec_.push_back(new Blob<Dtype>());
  }
  virtual ~SuperCategoryLabelLayerTest() { 
	  delete blob_bottom_label_;
	  for(auto it = blob_top_vec_.begin(); it != blob_top_vec_.end(); ++it)
		  delete *it;
  }
  Blob<Dtype>* const blob_bottom_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void SetSuperCategoryParam(SuperCategoryParameter * sup_param) {
	  SuperCategoryParameter::TreeScheme * root = sup_param->mutable_root();
	  SuperCategoryParameter::TreeScheme * child1 = root->add_children();
	  SuperCategoryParameter::TreeScheme * child2 = root->add_children();
	  SuperCategoryParameter::TreeScheme * child2_1 = child2->add_children();
	  SuperCategoryParameter::TreeScheme * child2_2 = child2->add_children();
	  SuperCategoryParameter::TreeScheme * child2_3 = child2->add_children();
	  SuperCategoryParameter::TreeScheme * child2_3_1 = child2_3->add_children();
	  SuperCategoryParameter::TreeScheme * child2_3_2 = child2_3->add_children();

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

	layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	EXPECT_EQ(this->blob_top_vec_[0]->cpu_data()[3], 1);
	EXPECT_EQ(this->blob_top_vec_[1]->cpu_data()[3], 3);
	EXPECT_EQ(this->blob_top_vec_[2]->cpu_data()[3], 3);

	layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
	EXPECT_EQ(this->blob_top_vec_[0]->cpu_data()[4], 1);
	EXPECT_EQ(this->blob_top_vec_[1]->cpu_data()[4], 3);
	EXPECT_EQ(this->blob_top_vec_[2]->cpu_data()[4], 4);
}

}  // namespace caffe
