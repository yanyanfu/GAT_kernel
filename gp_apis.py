import torch as th
import numpy as np
import torch.utils.dlpack
import kernel as gpk

# def gp_gspmm(g, X, dim0, dim1, inverse, norm):
#     X_dl = th.utils.dlpack.to_dlpack(X)

#     # declare the output tensor here
#     cuda0 = th.device('cuda')
#     res = th.zeros(dim0, dim1, device=cuda0)
#     res_dl = th.utils.dlpack.to_dlpack(res)

#     gpk.gspmm(g, X_dl, res_dl, inverse, norm)  # do not specify the reduce operation

#     return res

def gp_gspmm(s, d):

    s_row_dl = th.utils.dlpack.to_dlpack(s.crow_indices())
    s_col_dl = th.utils.dlpack.to_dlpack(s.col_indices())
    s_value_dl = th.utils.dlpack.to_dlpack(s.values())
    d_dl = th.utils.dlpack.to_dlpack(d.data)

    cuda0 = th.device('cuda')
    res = th.zeros(s.size()[0], d.size()[1], device=cuda0)
    res_dl = th.utils.dlpack.to_dlpack(res)

    gpk.gspmm(s_row_dl, s_col_dl, s_value_dl, d_dl, res_dl)

    return res