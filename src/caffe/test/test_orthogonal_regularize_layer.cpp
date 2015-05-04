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
class OrthogonalRegularizeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  OrthogonalRegularizeLayerTest()
	  : blob_bottom_0_(new Blob<Dtype>(50, 30, 1, 1)),
	    blob_bottom_1_(new Blob<Dtype>(10, 30, 1, 1)),
		blob_top_(new Blob<Dtype>(1,1,1,1))
  {
	  FillerParameter filler_param;
	  //filler_param.set_value(-1);
	  //ConstantFiller<Dtype> filler(filler_param);
	  UniformFiller<Dtype> filler(filler_param);
	  filler_param.set_min(-1);
	  filler_param.set_max(1);
	  //GaussianFiller<Dtype> filler(filler_param);
	  
	  filler.Fill(this->blob_bottom_0_);
	  filler.Fill(this->blob_bottom_1_);
	  blob_bottom_vec_.push_back(blob_bottom_0_);
	  blob_bottom_vec_.push_back(blob_bottom_1_);
	  blob_top_vec_.push_back(blob_top_);
  }

  virtual ~OrthogonalRegularizeLayerTest() {
	  delete blob_bottom_0_;
	  delete blob_bottom_1_;
	  delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(OrthogonalRegularizeLayerTest, TestDtypesAndDevices);
//TYPED_TEST_CASE(OrthogonalRegularizeLayerTest, DoubleCPU);

TYPED_TEST(OrthogonalRegularizeLayerTest, TestForward) {
	typedef typename TypeParam::Dtype Dtype;

	LayerParameter layer_param;
	
	shared_ptr<Layer<Dtype> > layer(
		new OrthogonalRegularizeLayer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

	const Dtype * data = this->blob_top_->cpu_data();
	std::cout << *data << std::endl;
}

TYPED_TEST(OrthogonalRegularizeLayerTest, TestBackward) {
	typedef typename TypeParam::Dtype Dtype;

	LayerParameter layer_param;
	
	shared_ptr<Layer<Dtype> > layer(
		new OrthogonalRegularizeLayer<Dtype>(layer_param));

    GradientChecker<Dtype> checker(1e-3, 1e-1);
    checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_,this->blob_top_vec_);
}

}
