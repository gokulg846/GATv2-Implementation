# GATv2-Implementation

This repository contains the code to the reimplementation of the paper - How Attentive are Graph Attention Networks (https://arxiv.org/abs/2105.14491)

The GATv2 model is tested on the QM-9 dataset and compared with the performance of GAT model that is available by default in Pytorch Geometric.

The model outputs the following graphs proving that GATv2 is in fact better at predicting the QM-9 dataset compared to GAT, confirming the results obtained in the original paper.

<img width="469" alt="image" src="https://github.com/user-attachments/assets/d4f952b4-bf6c-4924-ac97-b2f991eaf245"> <img width="469" alt="image" src="https://github.com/user-attachments/assets/d00dc957-26d4-4bd5-9033-2ae72ca9fbc4">



Requirements.txt file lists all the libraries used in Google Colab. Almost all are available by default. The only library that requires to be installed to run this code is Pytorch Geometric: !pip install torch_geometric.

