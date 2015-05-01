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

template <typename TypeParam>
class InnerProductWithRegularizeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  InnerProductWithRegularizeLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 5, 3, 3)),
        blob_top_(new Blob<Dtype>()) ,
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~InnerProductWithRegularizeLayerTest() { 
	  delete blob_bottom_; 
	  delete blob_top_; 
	  delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(InnerProductWithRegularizeLayerTest, TestDtypesAndDevices);

TYPED_TEST(InnerProductWithRegularizeLayerTest, TestSetup) {
	typedef typename TypeParam::Dtype Dtype;
	LayerParameter layer_param;
	InnerProductParameter* inner_product_param =
	  layer_param.mutable_inner_product_param();
	inner_product_param->set_num_output(2);

	shared_ptr<InnerProductLayer<Dtype> > layer(
	  new InnerProductWithRegularizeLayer<Dtype>(layer_param));
	layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	EXPECT_EQ(this->blob_top_->num(), 10);
	EXPECT_EQ(this->blob_top_->channels(), 2);
	EXPECT_EQ(this->blob_top_->height(), 1);
	EXPECT_EQ(this->blob_top_->width(), 1);
	EXPECT_EQ(this->blob_top_loss_->num(), 1);
	EXPECT_EQ(this->blob_top_loss_->height(), 1);
	EXPECT_EQ(this->blob_top_loss_->width(), 1);
	EXPECT_EQ(this->blob_top_loss_->channels(), 1);
	EXPECT_EQ(layer->blobs().size(),2);
}

TYPED_TEST(InnerProductWithRegularizeLayerTest, TestForward) {
	typedef typename TypeParam::Dtype Dtype;

    LayerParameter layer_param;
	//InnerProductParameter Setting
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(3);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);

    shared_ptr<Layer<Dtype> > layer(
        new InnerProductWithRegularizeLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
    }
	std::cout << *this->blob_top_vec_[1]->cpu_data() << std::endl;
}
TYPED_TEST(InnerProductWithRegularizeLayerTest, TestGradient) {
	typedef typename TypeParam::Dtype Dtype;

    LayerParameter layer_param;
	//InnerProductParameter Setting
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(3);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    inner_product_param->mutable_weight_filler()->set_min(1);
    inner_product_param->mutable_weight_filler()->set_max(2);
    inner_product_param->mutable_bias_filler()->set_type("uniform");
    inner_product_param->mutable_bias_filler()->set_min(1);
    inner_product_param->mutable_bias_filler()->set_max(2);

    shared_ptr<Layer<Dtype> > layer(
        new InnerProductWithRegularizeLayer<Dtype>(layer_param));
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(layer.get(), this->blob_bottom_vec_, this->blob_top_vec_);
}
}
