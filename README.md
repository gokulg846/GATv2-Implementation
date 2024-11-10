# GATv2-Implementation

This repository contains the code to the reimplementation of the paper - How Attentive are Graph Attention Networks (https://arxiv.org/abs/2105.14491)

The GATv2 model is tested on the QM-9 dataset and compared with the performance of GAT model that is available by default in Pytorch Geometric.

The model outputs the following graphs proving that GATv2 is in fact better at predicting the QM-9 dataset compared to GAT, confirming the results obtained in the original paper. (Refer the images added)







Requirements.txt file lists all the libraries used in Google Colab. Almost all are available by default. The only library that requires to be installed to run this code is Pytorch Geometric: !pip install torch_geometric.

