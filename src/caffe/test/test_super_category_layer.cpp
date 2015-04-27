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
class SuperCategoryLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SuperCategoryLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(3, 5, 1, 1))
  {
    // fill the values for bottom
    FillerParameter filler_param;
	filler_param.set_min(1);
	filler_param.set_max(2);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
	//add to vector
    blob_bottom_vec_.push_back(blob_bottom_data_);

	// make the top
	for(int i = 0; i < 3 ; ++i)
		blob_top_vec_.push_back(new Blob<Dtype>());
  }
  virtual ~SuperCategoryLayerTest() { 
	  delete blob_bottom_data_;
	  for(auto it = blob_top_vec_.begin(); it != blob_top_vec_.end(); ++it)
		  delete *it;
  }
  Blob<Dtype>* const blob_bottom_data_;
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

TYPED_TEST_CASE(SuperCategoryLayerTest, TestDtypesAndDevices);

TYPED_TEST(SuperCategoryLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  this->SetSuperCategoryParam(layer_param.mutable_super_category_param());

  shared_ptr<SuperCategoryLayer<Dtype> > layer(
      new SuperCategoryLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_vec_[0]->num(),3);
  EXPECT_EQ(this->blob_top_vec_[0]->channels(),2);
  EXPECT_EQ(this->blob_top_vec_[0]->height(),1);
  EXPECT_EQ(this->blob_top_vec_[0]->width(),1);
  EXPECT_EQ(this->blob_top_vec_[1]->num(),3);
  EXPECT_EQ(this->blob_top_vec_[1]->channels(),4);
  EXPECT_EQ(this->blob_top_vec_[1]->height(),1);
  EXPECT_EQ(this->blob_top_vec_[1]->width(),1);
  EXPECT_EQ(this->blob_top_vec_[2]->num(),3);
  EXPECT_EQ(this->blob_top_vec_[2]->channels(),5);
  EXPECT_EQ(this->blob_top_vec_[2]->height(),1);
  EXPECT_EQ(this->blob_top_vec_[2]->width(),1);
}

TYPED_TEST(SuperCategoryLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
	sizeof(Dtype) == 4 || IS_VALID_CUDA) {
	LayerParameter layer_param;
	this->SetSuperCategoryParam(layer_param.mutable_super_category_param());

	shared_ptr<SuperCategoryLayer<Dtype> > layer(
	  new SuperCategoryLayer<Dtype>(layer_param));
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

TYPED_TEST(SuperCategoryLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {

	LayerParameter layer_param;
	this->SetSuperCategoryParam(layer_param.mutable_super_category_param());

	SuperCategoryLayer<Dtype> * layer = new SuperCategoryLayer<Dtype>(layer_param);

    GradientChecker<Dtype> checker(1e-4, 1e-3);
    checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_,
        this->blob_top_vec_);

	delete layer;
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
