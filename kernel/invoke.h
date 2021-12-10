#pragma once

#include "kernel.h"

void invoke_spmm(int64_t * s_row, int64_t * s_col, float * s_value, float * d, float * y, int64_t v_count, int64_t dim);

// void invoke_spmm(csr_t*  obj1, array2d_t < float >&  x1, array2d_t < float >&  y1, op_t op, bool reverse, bool norm, int dim);
// graph_t* invoke_init_graph(vid_t v_count, vid_t dst_size, vid_t* offset_csr, void* nebrs_csr, vid_t* offset_csc, void* nebrs_csc);
