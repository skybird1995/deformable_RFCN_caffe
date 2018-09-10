//
//---------------------------------------------
//Written by jiuhong chen



#include <cfloat>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/deformable_psroi_pooling.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe{
	
	template <typename Dtype>	
	void DeformablePSROIPoolLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top) 
	{
		DeformableRSROIParameter deformable_peroi_param=this->layer_param_.deformable_peroi_param();
		spatial_scale_=deformable_peroi_param.spatial_scale();
		
		LOG(INFO) << "Spatial scale: " << spatial_scale_;


		CHECK_GT(psroi_align_param.output_dim(), 0)
	      	<< "output_dim must be > 0";
		CHECK_GT(psroi_align_param.group_size(), 0)
	      	<< "group_size must be > 0";
		
		output_dim_ = deformable_peroi_param.output_dim();
		group_size_ = deformable_peroi_param.group_size();
		
		sample_per_part = deformable_peroi_param.sample_num();
			
		pooled_height_ = group_size_;
		pooled_width_ = group_size_;
		
		part_size_ = group_size_;
		
	}

	template <typename Dtype>
	void DeformablePSROIPoolLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
	{
		channels_ = bottom[0]->channels();
		
    	CHECK_EQ(channels_, output_dim_*group_size_*group_size_)
			  << "input channel number does not match layer parameters";
		
		height_= bottom[0]->height();
		width_ = bottom[0]->width();
		
		top[0]->Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
		
		//mapping_channel_.Reshape(bottom[1]->num(), output_dim_, pooled_height_, pooled_width_);
    	//sample_pos_.Reshape(bottom[1]->num(), output_dim_, pooled_height_*pooled_width_*sample_num_*sample_num_, 2);
	}
	
	template <typename Dtype>
	void DeformablePSROIPoolLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
	{
		NOT_IMPLEMENTED;
	}

	
	template <typename Dtype>
	void DeformablePSROIPoolLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
	{
		NOT_IMPLEMENTED;
	}
	
#ifdef_CPU_ONLY
	STUB_GPU(DeformablePSROIPoolLayer);
#endif
  INSTANTIATE_CLASS(DeformablePSROIPoolLayer);
  REGISTER_LAYER_CLASS(DeformablePSROIPool);	
	
}
