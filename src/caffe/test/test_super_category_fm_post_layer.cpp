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
class SuperCategoryFMPostLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SuperCategoryFMPostLayerTest()
  {
	//add to vector
	

	// make the top
	for(int i = 0; i < 3 ; ++i)
	{
		blob_bottom_vec_.push_back(new Blob<Dtype>());
		blob_top_vec_.push_back(new Blob<Dtype>());
	}

	// make the top
	for(int i = 0; i < 5 ; ++i)
	{
		blob_bottom_vec_hard_.push_back(new Blob<Dtype>());
		blob_top_vec_hard_.push_back(new Blob<Dtype>());
	}
  }
  virtual ~SuperCategoryFMPostLayerTest() { 
	  for(auto it = blob_bottom_vec_.begin(); it != blob_bottom_vec_.end(); ++it)
		  delete *it;
	  for(auto it = blob_bottom_vec_hard_.begin(); it != blob_bottom_vec_hard_.end(); ++it)
		  delete *it;
	  for(auto it = blob_top_vec_.begin(); it != blob_top_vec_.end(); ++it)
		  delete *it;
	  for(auto it = blob_top_vec_hard_.begin(); it != blob_top_vec_hard_.end(); ++it)
		  delete *it;
  }
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_hard_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_top_vec_hard_;

  void SetSuperCategoryParam(SuperCategoryParameter * sup_param) {
	  this->blob_bottom_vec_[0]->Reshape(2,2,2,2);
	  this->blob_bottom_vec_[1]->Reshape(2,4,2,2);
	  this->blob_bottom_vec_[2]->Reshape(2,5,2,2);
	  // fill the values for bottom
	  FillerParameter filler_param;
	  filler_param.set_min(1);
	  filler_param.set_max(2);
	  UniformFiller<Dtype> filler(filler_param);
	  for(int i = 0; i < 3; ++i)
		filler.Fill(this->blob_bottom_vec_[i]);
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
	  this->blob_bottom_vec_hard_[0]->Reshape(2,2,2,2);
	  this->blob_bottom_vec_hard_[1]->Reshape(2,5,2,2);
	  this->blob_bottom_vec_hard_[2]->Reshape(2,8,2,2);
	  this->blob_bottom_vec_hard_[3]->Reshape(2,12,2,2);
	  this->blob_bottom_vec_hard_[4]->Reshape(2,13,2,2);
	  // fill the values for bottom
	  FillerParameter filler_param;
	  filler_param.set_min(1);
	  filler_param.set_max(2);
	  UniformFiller<Dtype> filler(filler_param);
	  for(int i = 0; i < 5; ++i)
		filler.Fill(this->blob_bottom_vec_hard_[i]);
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

TYPED_TEST_CASE(SuperCategoryFMPostLayerTest, TestDtypesAndDevices);

TYPED_TEST(SuperCategoryFMPostLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  this->SetSuperCategoryParam(layer_param.mutable_super_category_param());

  shared_ptr<SuperCategoryFMPostLayer<Dtype> > layer(
      new SuperCategoryFMPostLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  for(int i = 0; i < 3; ++i)
  {
	  EXPECT_EQ(this->blob_top_vec_[i]->num(),this->blob_bottom_vec_[i]->num());
	  EXPECT_EQ(this->blob_top_vec_[i]->channels(),this->blob_bottom_vec_[i]->channels());
	  EXPECT_EQ(this->blob_top_vec_[i]->width(),this->blob_bottom_vec_[i]->width());
	  EXPECT_EQ(this->blob_top_vec_[i]->height(),this->blob_bottom_vec_[i]->height());
  }
}

TYPED_TEST(SuperCategoryFMPostLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
	sizeof(Dtype) == 4 || IS_VALID_CUDA) {
	LayerParameter layer_param;
	layer_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
	this->SetSuperCategoryParam(layer_param.mutable_super_category_param());

	shared_ptr<SuperCategoryFMPostLayer<Dtype> > layer(
	  new SuperCategoryFMPostLayer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

	for(int i = 0; i < 3; ++i ) {
		const Dtype* data = this->blob_top_vec_[i]->cpu_data();
		const int count = this->blob_top_vec_[i]->count();
		for (int i = 0; i < count; ++i) {
			EXPECT_GE(data[i], 1.);
		}
	}
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}
TYPED_TEST(SuperCategoryFMPostLayerTest, TestForward_MINUS) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
	sizeof(Dtype) == 4 || IS_VALID_CUDA) {
	LayerParameter layer_param;
	layer_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_MINUS);
	this->SetSuperCategoryParam(layer_param.mutable_super_category_param());

	shared_ptr<SuperCategoryFMPostLayer<Dtype> > layer(
	  new SuperCategoryFMPostLayer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

	for(int i = 1; i < 3; ++i ) {
		const Dtype* data = this->blob_top_vec_[i]->cpu_data();
		const int count = this->blob_top_vec_[i]->count();
		for (int i = 0; i < count; ++i) {
			EXPECT_LE(data[i], 1.);
		}
	}
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}
TYPED_TEST(SuperCategoryFMPostLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

	LayerParameter layer_param;
	layer_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
	this->SetSuperCategoryParam(layer_param.mutable_super_category_param());

	SuperCategoryFMPostLayer<Dtype> * layer = new SuperCategoryFMPostLayer<Dtype>(layer_param);

    GradientChecker<Dtype> checker(1e-4, 1e-2);
    checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_,
        this->blob_top_vec_);

	delete layer;
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(SuperCategoryFMPostLayerTest, TestGradient_Hard) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

	LayerParameter layer_param;
	layer_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
	this->SetSuperCategoryParam_Hard(layer_param.mutable_super_category_param());

	SuperCategoryFMPostLayer<Dtype> * layer = new SuperCategoryFMPostLayer<Dtype>(layer_param);

    GradientChecker<Dtype> checker(1e-4, 1e-2);
    checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_hard_,
        this->blob_top_vec_hard_);

	delete layer;
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}
TYPED_TEST(SuperCategoryFMPostLayerTest, TestGradient_MINUS) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

	LayerParameter layer_param;
	layer_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_MINUS);
	this->SetSuperCategoryParam(layer_param.mutable_super_category_param());

	SuperCategoryFMPostLayer<Dtype> * layer = new SuperCategoryFMPostLayer<Dtype>(layer_param);

    GradientChecker<Dtype> checker(1e-4, 1e-2);
    checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_,
        this->blob_top_vec_);

	delete layer;
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(SuperCategoryFMPostLayerTest, TestGradient_Hard_MINUS) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

	LayerParameter layer_param;
	layer_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_MINUS);
	this->SetSuperCategoryParam_Hard(layer_param.mutable_super_category_param());

	SuperCategoryFMPostLayer<Dtype> * layer = new SuperCategoryFMPostLayer<Dtype>(layer_param);

    GradientChecker<Dtype> checker(1e-4, 1e-2);
    checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_hard_,
        this->blob_top_vec_hard_);

	delete layer;
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}
}  // namespace caffe

