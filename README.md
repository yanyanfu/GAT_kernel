# Pytorch Graph Attention Network

This is an implementation of the Graph Attention Network (GAT) based on spmm calculation in Kernel
model presented by Veličković et. al (2017, https://arxiv.org/abs/1710.10903).

The repo is based on the pytorch implementation of GAT https://github.com/Diego999/pyGAT. The official repository for the GAT (Tensorflow) is available in https://github.com/PetarV-/GAT. Therefore, if you make advantage of the GAT-kernel model in your research, please cite the following:

```
@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
  note={accepted as poster},
}
```


# Steps

```
git clone --recurse-submodules https://github.com/yanyanfu/GAT_kernel
cd GAT_kernel
```

```
cd kernel
make
cd ..
cmake kernel
make
```

```
python3 train.py
```


# Sparse version GAT

Based on the original repo, the sparse version GAT using pytorch are numerically instability because of softmax function. 

# Requirements

GAT-kernel relies on Python 3.5 and PyTorch 0.4.1 (due to torch.sparse_coo_tensor).


