#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "kernel.h"
#include "invoke.h"
#include <cassert>
#include <iostream>
#include <limits>
#define SIZE 1024
#define Block 1
#define FULL_WARP_MASK 0xFFFFFFFF


template <class T>
__device__ T warp_reduce(T val){
    for(int offset=32/2;offset>0;offset/=2)
        val+= __shfl_down_sync (FULL_WARP_MASK,val,offset);
    return val;

}

typedef float (*op_scalar_fn)(float, float);

__device__ inline float add_scalar(float x, float y) {
    return x + y;
}

__device__ inline float sub_scalar(float x, float y) {
    return x - y;
}

__device__ inline float max_scalar(float x, float y) {
    if(x>y) return x;
    else return y;
}

__device__ inline float min_scalar(float x, float y) {
    if(x<y) return x;
    else return y;
}

__device__ inline float mul_scalar(float x, float y) {
    return x * y;
}

__device__ inline float div_scalar(float x, float y) {
    return x / y;
}

__device__ op_scalar_fn  p_mul = mul_scalar;
__device__ op_scalar_fn  p_div = div_scalar;
__device__ op_scalar_fn  p_add = add_scalar;
__device__ op_scalar_fn  p_sub = sub_scalar;
__device__ op_scalar_fn  p_min = min_scalar;
__device__ op_scalar_fn  p_max = max_scalar;

//to be used if host is sending function pointer to kernel
inline op_scalar_fn get_fn(op_t op) {
    op_scalar_fn op_fn;

    if (op == eDIV) {
        cudaMemcpyFromSymbol(&op_fn, p_div, sizeof(op_scalar_fn));
        //op_fn = div_scalar;
    } else if (op == eSUB) {
        cudaMemcpyFromSymbol(&op_fn, p_sub, sizeof(op_scalar_fn));
        //op_fn = sub_scalar;
    } else if (op == eSUM) {
        cudaMemcpyFromSymbol(&op_fn, p_add, sizeof(op_scalar_fn));
        //op_fn = add_scalar;
    } else if (op == eMUL) {
        cudaMemcpyFromSymbol(&op_fn, p_mul, sizeof(op_scalar_fn));
        //op_fn = mul_scalar;
    } else if (op == eMIN) {
        cudaMemcpyFromSymbol(&op_fn, p_min, sizeof(op_scalar_fn));
        //op_fn = min_scalar;
    } else if (op == eMAX) {
        cudaMemcpyFromSymbol(&op_fn, p_max, sizeof(op_scalar_fn));
        //op_fn = max_scalar;
    } else {
        assert(0);
    }
    return op_fn;
}

//if the kernel itself need the fuction pointer
__device__ inline op_scalar_fn get_fn_kernel(op_t op) {
    op_scalar_fn op_fn;

    if (op == eDIV) {
        op_fn = div_scalar;
    } else if (op == eSUB) {
        op_fn = sub_scalar;
    } else if (op == eSUM) {
        op_fn = add_scalar;
    } else if (op == eMUL) {
        op_fn = mul_scalar;
    } else if (op == eMIN) {
        op_fn = min_scalar;
    } else if (op == eMAX) {
        op_fn = max_scalar;
    } else {
        assert(0);
    }
    return op_fn;
}

__global__ void spmm(int64_t * s_row, int64_t * s_col, float * s_value, float * d, float * y, const int64_t v_count, const int64_t dim) 
{
    //TODO
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= v_count)
        return;

    float * output = y + dim * row;
    for (int i = 0; i < dim; i++){
        for (int j = s_row[row]; j < s_row[row+1]; j++){
            float * row_d = d + dim * s_col[j];
            output[i] += s_value[j] * row_d [i];
        }
    }
}

void invoke_spmm(int64_t * s_row, int64_t * s_col, float * s_value, float * d, float * y, const int64_t v_count, const int64_t dim)
{
    //int warp_size=32; 
    int block_size = 1024;
    int nBlocks = ceil(v_count/(float)block_size); // TODO 
    //spmm_warp <<<nBlocks,block_size>>> (obj1, x1.data_ptr, y1.data_ptr, op, true, true, dim);
    //spmm <<<nBlocks,block_size>>> (obj1, x1.data_ptr, y1.data_ptr, op, reverse, norm, dim); 
    spmm <<<nBlocks, block_size>>> (s_row, s_col, s_value, d, y, v_count, dim);
    cudaDeviceSynchronize();
}

//warp per row (best)
//__global__ void spmm_warp(const csr_t* __restrict__ obj1, float* x, float * y, op_t op, const bool reverse, const bool norm, const int dim)
// {
//     //TODO

// }

// graph_t * invoke_init_graph(vid_t v_count, vid_t dst_size, vid_t * offset_csr, void * nebrs_csr, vid_t * offset_csc, void * nebrs_csc) {

//     //Let us make a cpu graph first
//     graph_t g;
//     g.init_cpu(v_count, dst_size, 
//             offset_csr, nebrs_csr,
//             offset_csc, nebrs_csc);

//     graph_t * graph = (graph_t*) malloc(sizeof(graph_t));
//     cudaMallocManaged( & graph->csr,  sizeof(csr_t));

//     vid_t edge_count = offset_csr[v_count];
//     vid_t * offset_csr_gpu;
//     vid_t * offset_csc_gpu;
//     char * nebrs_csr_gpu;
//     char * nebrs_csc_gpu;

//     cudaMallocManaged( & offset_csr_gpu, (v_count + 1) * sizeof(vid_t));
//     cudaMallocManaged( & nebrs_csr_gpu, edge_count * dst_size);

//     //memcopy
//     cudaMemcpy(offset_csr_gpu, offset_csr, (v_count + 1) * sizeof(vid_t), cudaMemcpyHostToDevice);
//     cudaMemcpy(nebrs_csr_gpu, nebrs_csr, edge_count * dst_size, cudaMemcpyHostToDevice);


//     if (nebrs_csr == nebrs_csc) {
//         graph->csc = graph->csr;
//         offset_csc_gpu = offset_csr_gpu;
//         nebrs_csc_gpu = nebrs_csr_gpu;
//     } else {
//         cudaMallocManaged( & graph->csc,  sizeof(csr_t));
//         cudaMallocManaged( & offset_csc_gpu, (v_count + 1) * sizeof(vid_t));
//         cudaMallocManaged( & nebrs_csc_gpu, edge_count * dst_size);

//         cudaMemcpy(nebrs_csc_gpu, nebrs_csc, edge_count * dst_size, cudaMemcpyHostToDevice);
//         cudaMemcpy(offset_csc_gpu, offset_csc, (v_count + 1) * sizeof(vid_t), cudaMemcpyHostToDevice);
//     }

//     //printf("invoke init graph called\n");
//     graph -> init(v_count, dst_size, offset_csr_gpu, nebrs_csr_gpu, offset_csc_gpu, nebrs_csc_gpu);

//     return graph;

// }

