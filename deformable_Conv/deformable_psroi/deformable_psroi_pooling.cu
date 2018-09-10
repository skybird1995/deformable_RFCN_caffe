//-----------------------------------
//Written by jiuhong chen
//----------------------------------


#include <algorithm>
#include <cfloat>
#include <vector>
#include <stdio.h>

#include "caffe/layers/deformable_psroi_pooling.hpp"

#include "caffe/util/gpu_util.cuh"

using std::max;
using std::min;

namespace caffe {

	template<typename Dtype>
	__device__ void bilinear_interpolate(
		const DType* data,
		const DType x,
		const DTupe y,
		const int width,
		const int height)
	{
	    int x1 = floor(x);
		int x2 = ceil(x);
		
		int y1 = floor(y);
		int y2 = ceil(y);
		
		DType dist_x = static_cast<DType>(x-x1);
		DType dist_y = static_cast<DType>(y-y1);
		
		DType value11 = data[y1*width + x1];
      	DType value12 = data[y2*width + x1];
      	DType value21 = data[y1*width + x2];
      	DType value22 = data[y2*width + x2];
		
		DType value = (1 - dist_x)*(1 - dist_y)*value11 + (1 - dist_x)*dist_y*value12
        	+ dist_x*(1 - dist_y)*value21 + dist_x*dist_y*value22;
		
		return value;
	}
	
	
	template <typename Dtype>
	__global__ void DeformablePSROIPoolForward(
	const int nthreads,
	const Dtype* bottom_data,
	const Dtype spatial_scale,
	const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois,
	const Dtype* bottom_trans,
	const bool no_trans
	const Dtype trans_std,
	const int sample_per_part,
    const int output_dim,  // 输出通道数
    const int group_size,  // k*k*(c+1) 中的 k
	
	const int part_szie,

	const int num_classes,
	const int channels_each_class,
    Dtype* top_data)
	{
		CUDA_KERNEL_LOOP(index,nthreads)
		{
			//The output is in order (n, ctop, ph, pw)
			int pw = index % pooled_width;
			int ph = (index /pooled_width) % pooled_height;
			
			int ctop = (index / pooled_width / pooled_height) % output_dim;
			
			int n = (index/pooled_width/pooled_height) / output_dim;
			
			// [start, end) interval for spatial sampling
			bottom_rios = bottom_rois +n*5;
			int roi_batch_ind = bottom_rios[0];
			Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1]))*spatial_scale-0.5;
			Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2]))*spatial_scale-0.5;
			Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3])+1.)*spatial_scale-0.5;
			Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4])+1.)*spatial_scale-0.5;
			
			// Force too small ROIS to be 1*1
			Dtype roi_width= max(roi_end_w-roi_start_w, 0.1);  //avoid 0
			Dtype roi_height= max(roi_end_h-roi_start_h, 0.1);  //avoid 0
			
			//Compute w and h at bottom 
			Dtype bin_size_h = roi_height/static_cast<Dtype>(pooled_height);
			Dtype bin_size_w = roi_width/static_cast<Dtype>(pooled_width);
			
			Dtype sub_bin_size_h = bin_size_h / static_cast<Dtype>(sample_per_part);
        	Dtype sub_bin_size_w = bin_size_w / static_cast<Dtype>(sample_per_part);
			
			int part_h = floor(static_cast<Dtype>(ph) / pooled_height*part_size);
        	int part_w = floor(static_cast<Dtype>(pw) / pooled_width*part_size);
			
			int class_id = ctop / channels_each_class;
			
			Dtype trans_x= no_trans ? static_cast<Dtype>(0) :
          		bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h)*part_size + part_w] * trans_std;
			Dtype trans_y = no_trans ? static_cast<Dtype>(0) :
          		bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h)*part_size + part_w] * trans_std;
			
			Dtype wstart = static_cast<Dtype>(pw)* bin_size_w+ roi_start_w;
			wstart += trans_x * roi_width;
			
			Dtype hstart = static_cast<Dtype>(ph) * bin_size_h+ roi_start_h;
			hstart += trans_y * roi_height;
			
			Dtype sum = 0;
        	int count = 0;
			
			int gw = floor(static_cast<Dtype>(pw) * group_size / pooled_width);
			int gh = floor(static_cast<Dtype>(ph)* group_size / pooled_height);
			gw = min(max(gw, 0), group_size - 1);
			gh = min(max(gh, 0), group_size - 1);
			
			const Dtype* offset_bottom_data = bottom_data + (roi_batch_ind * channels) * height * width;
			
			for(int ih=0; ih<sample_per_part; ih++)
			{
				for(iw=0; iw<sample_per_part; iw++)
				{
					Dtype w = wstart + iw*sub_bin_size_w;
					Dtype h = hstart + ih*sub_bin_size_h;
					
					int c = (ctop*group_size + gh)*group_size +gw;
					
					Dtype val = bilinear_interp(offset_bottom_data + c*height*width, w, h, width, height);
					
					sum += val;
            		count++;
				}
			}
			top_data[index] = count==0 ? static_cast<Dtype>(0) : sum/count;
			//top_count[index] = count;
		}
	}
	
	
	
	template<typename Dtype>
	void DeformablePSROIPoolLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
	{
	   const Dtype* bottom_data=bottom[0]->gpu_data();
	   const Dtype* bottom_rois=bottom[1]->gpu_data();
	   
	   Dtype* top_data = top[0]->mutable_gpu_data();
	   //Dtype* sample_pos_data = sample_pos_.mutable_gpu_data();
       //int* mapping_channel_ptr = mapping_channel_.mutable_gpu_data();
	   Dtype* bottom_trans = NULL;
	   
	   
	   int no_trans = Ture;
	   int num_classes = 0;
	   int channels_each_class = 0;
	   
	   const float trans_std = 0.1;
	   
	   int count=top[0]->count();
	   caffe_gpu_set(count, Dtype(0), top_data);
	   //caffe_gpu_set(count, -1, mapping_channel_ptr);
	   
	   
	   if(bottom.size()>=3)
	   {
			no_trans = False;
			num_classes = bottom[2]->channels() / 2;
			channels_each_class = output_dim_ / num_classes;
			
			bottom_trans =  bottm[2]->gpu_data();
			
	   }else
	   {
	   		num_classes = 1;
	   		no_trans = True;
			channels_each_class = output_dim_;
			
			bottom_trans = NULL;
	   }
	   
	   
	   
	   
	   DeformablePSROIPoolForward<Dtype> << <CAFFE_GET_BLOCKS(count),
	   CAFFE_CUDA_NUM_THREADS >> >(count, bottom_data, spatial_scale_, channels_, height_, width_, pooled_height_, pooled_width_,
	   bottom_rois, bottom_trans, no_trans, trans_std, sample_per_part, output_dim_, group_size_, part_size_, num_classes, channels_each_class,
	   top_data);
	   
	   CUDA_POST_KERNEL_CHECK;
	   
	}
	
	
	template <typename Dtype>
	__global__ void DeformablePSROIPoolBackwardAtomic(
		const int nthreads,
		const Dtype* top_diff,
		const Dtype spatial_scale,
		
		const int channels,
		const int height, const int width,
		const int pooled_height, const int pooled_width,
		const int output_dim,
		
		Dtype* bottom_data_diff, Dtype* bottom_trans_diff,
		const Dtype* bottom_data,
		const Dtype* bottom_rois,
		const Dtype* bottom_trans,
		
		const bool no_trans,
		const Dtype trans_std,
		const int sample_per_part,
		const int group_size,
		const int part_size,
		
		const int num_classes,
		const int channels_each_class)
	 {
		CUDA_KERNEL_LOOP(index, nthreads){
			// The output is in order (n, ctop, ph, pw)
			int pw = index % pooled_width;
			int ph = (index / pooled_width) % pooled_height;
			int ctop = (index / pooled_width /pooled_height) % output_dim;
			
			int n = (index / pooled_width / pooled_height)/ output_dim;
			
			bottom_rois = bottom_rois + 5*n;
			int roi_batch_ind = bottom_roi[0];
			
			Dtype roi_start_w = static_cast<Dtype>(round(bottom_rois[1])) * spatial_scale - 0.5;
        	Dtype roi_start_h = static_cast<Dtype>(round(bottom_rois[2])) * spatial_scale - 0.5;
        	Dtype roi_end_w = static_cast<Dtype>(round(bottom_rois[3]) + 1.) * spatial_scale - 0.5;
        	Dtype roi_end_h = static_cast<Dtype>(round(bottom_rois[4]) + 1.) * spatial_scale - 0.5;
			
			//Force too small ROIS to be 1*1
			Dtype roi_width = max(roi_end_w - roi_start_w, 0.1); //avoid 0
        	Dtype roi_height = max(roi_end_h - roi_start_h, 0.1);
			
			// Compute w and h at bottom
			Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
        	Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);
			
			Dtype sub_bin_size_h = bin_size_h / static_cast<Dtype>(sample_per_part);
        	Dtype sub_bin_size_w = bin_size_w / static_cast<Dtype>(sample_per_part);
			
			
			int part_h = floor(static_cast<Dtype>(ph) / pooled_height*part_size);
        	int part_w = floor(static_cast<Dtype>(pw) / pooled_width*part_size);
			
			int class_id = ctop / channels_each_class;
		  	Dtype trans_x = no_trans ? static_cast<Dtype>(0) :
          		bottom_trans[(((n * num_classes + class_id) * 2) * part_size + part_h)*part_size + part_w] * trans_std;
			
			Dtype trans_y = no_trans ? static_cast<Dtype>(0) :
          		bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size + part_h)*part_size + part_w] * trans_std;
			
			Dtype wstart = static_cast<Dtype>(pw)* bin_size_w
          		+ roi_start_w;
        	wstart += trans_x * roi_width;
			
        	Dtype hstart = static_cast<Dtype>(ph) * bin_size_h
          		+ roi_start_h;
        	hstart += trans_y * roi_height;
			
			//if(top_count[index] <=0)
			//{ conyinue; }
			
			Dtype diff_val = top_diff[index] / (sample_per_part*sample_per_part);
			const Dtype offset_bottom_data = bottom_data + roi_batch_ind*channels*height*width;
			
			Dtype* offset_bottom_data_diff = bottom_data_diff + roi_batch_ind*channels*height*width;
			
			int gw = floor(static_cast<Dtype>(pw)* group_size / pooled_width);
        	int gh = floor(static_cast<Dtype>(ph)* group_size / pooled_height);
        	gw = min(max(gw, 0), group_size - 1);
        	gh = min(max(gh, 0), group_size - 1);
			
			for (int ih=0; ih<sample_per_part; ih++ )
			{
				for(int iw=0; iw<sample_per_part; iw++)
				{
										Dtype w = wstart + iw*sub_bin_size_w;
					Dtype h = hstart + ih*sub_bin_size_h;
					
					w = min(max(w, 0.), width - 1.);
            		h = min(max(h, 0.), height - 1.);
					
					int c = (ctop*group_size + gh)*group_size + gw;
					
					//Backwork on feature
					int x0 = floor(w);
					int x1 = ceil(w);
					int y0 = floor(h);
					int y1 = ceil(h);
					Dtype dist_x = w - x0;
					Dtype dist_y = h - y0;
					
					Dtype q00 = (1-dist_x)*(1-dist_y);
					Dtype q01 = (1-dist_x)*dist_y;
					Dtype q10 = dist_x*(1-dist_y);
					Dtype q11 = dist_x*dist_y;
					
					int bottom_index_base = c*height*width;
					
					caffe_gpu_atomic_add(q00*diff_val, offset_bottom_data_diff +  bottom_index_base + y0*width + x0);
					caffe_gpu_atomic_add(q01*diff_val, offset_bottom_data_diff +  bottom_index_base + y1*width + x0);
					caffe_gpu_atomic_add(q10*diff_val, offset_bottom_data_diff +  bottom_index_base + y0*width + x1);
					caffe_gpu_atomic_add(q11*diff_val, offset_bottom_data_diff +  bottom_index_base + y1*width + x1);
					
					if(no_trans)
					{ continue;}
					
					
					Dtype U00 = offset_bottom_data[bottom_index_base + y0*width + x0];
            		Dtype U01 = offset_bottom_data[bottom_index_base + y1*width + x0];
           		 	Dtype U10 = offset_bottom_data[bottom_index_base + y0*width + x1];
            		Dtype U11 = offset_bottom_data[bottom_index_base + y1*width + x1];
					
					Dtype diff_x = (U11*dist_y + U10*(1 - dist_y) - U01*dist_y - U00*(1 - dist_y))*trans_std*diff_val;
            		diff_x *= roi_width;
			
            		Dtype diff_y = (U11*dist_x + U01*(1 - dist_x) - U10*dist_x - U00*(1 - dist_x))*trans_std*diff_val;
            		diff_y *= roi_height;
					
					
					caffe_gpu_atomic_add(diff_x, bottom_trans_diff + (((n * num_classes + class_id) * 2) * part_size + part_h)*part_size + part_w);
					caffe_gpu_atomic_add(diff_y, bottom_trans_diff + (((n * num_classes + class_id) * 2 + 1)*part_size + part_h)*part_size + part_w);
				}

				
			}
		}
    }
	
	
	
	template <typename Dtype>
	void DeformablePSROIPoolLayer<Dtype>:: Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		if(!propagate_down[0])
			return 0;
		
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* bottom_rois = bottom[1]->gpu_data();
	    const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_data_diff = bottom[0]->mutable_gpu_diff();
		Dtype* bottom_trans=NULL;
		Dtype* bottom_trans_diff=NULL;
		
		
		int num_classes=0;
		int channels_each_class=0;
		const float trans_std=0.1;
		bool no_trans= Ture;
		
		if(bottom.size()>=3)
		{
			 bottom_trans = bottom[2]->gpu_data();
			 bottom_trans_diff = bottom[2]->mutable_gpu_diff();
			 
			 caffe_gpu_set(bottom[2]->count(), Dtype(0), bottom[2]->mutable_gpu_diff());  // offset_diff
			
			num_classes = bottom[2]->channels() / 2;
			channels_each_class = output_dim_/num_classes;
			
			no_trans=False;
			
		
		}else
		{
			bottom_trans = NULL;
			bottom_trans_diff = NULL;
			num_classes=1;
			channels_each_class=output_dim_;
			
			no_trans=Ture;
		}
		
		
		const int bottom_count = bottom[0]->count();
		
		caffe_gpu_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_gpu_diff());  // rois_diff
		caffe_gpu_set(bottom_count, Dtype(0), bottom_data_diff);                          // feature_map
		
		const int count = top[0]->count();
		
		DeformablePSROIPoolBackwardAtomic<Dtype> << < CAFFE_GET_BLOCKS(count),
			CAFFE_CUDA_NUM_THREADS >> >(
			count , top_diff , spatial_scale_, channel_, height_, width_, 
			pooled_height_, pooled_width, output_dim_, bottom_data_diff, bottom_trans_diff, 
			bottom_data, bottom_rios, bottom_trans, no_trans ,trans_std, sample_per_part_, 
			group_size_, part_size_, num_classes, channels_each_class);
			
		CUDA_POST_KERNEL_CHECK;
		
	}
	
	
	
	
}