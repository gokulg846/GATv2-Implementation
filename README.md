# GATv2-Implementation

This repository contains the code to the reimplementation of the paper - How Attentive are Graph Attention Networks (https://arxiv.org/abs/2105.14491)

The GATv2 model is tested on the QM-9 dataset and compared with the performance of GAT model that is available by default in Pytorch Geometric.

The model outputs the following graphs proving that GATv2 is in fact better at predicting the QM-9 dataset compared to GAT, confirming the results obtained in the original paper.
![image](https://github.com/user-attachments/assets/165b2bbf-4859-4473-ab27-27fc41b19e53)![image](https://github.com/user-attachments/assets/00b19c08-71ea-4aa8-a7d7-8e9fd8349d35)
![image](https://github.com/user-attachments/assets/fbbd9a22-60c6-4f23-afaf-49d61009fa36)







Requirements.txt file lists all the libraries used in Google Colab. Almost all are available by default. The only library that requires to be installed to run this code is Pytorch Geometric: !pip install torch_geometric.

