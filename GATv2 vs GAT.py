!pip install torch_geometric
import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
def plot_attention_comparison(attention1, attention2, figsize=(20, 8)):
    """
    Plot two attention heatmaps side by side.
    
    Args:
        attention1: First attention matrix (query vs key)
        attention2: Second attention matrix (query vs key)
        figsize: Size of the figure (width, height)
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # First heatmap
    sns.heatmap(attention1[:10], 
                ax=ax1,
                cmap='RdPu',  # Red-Purple colormap similar to the image
                fmt='.2f',    # Show 2 decimal places
                annot=True,   # Show numbers in cells
                square=True,  # Make cells square
                cbar=True,
                vmin=0,
                vmax=1,
                annot_kws={'size': 8})
    
    # Second heatmap
    sns.heatmap(attention2[:10], 
                ax=ax2,
                cmap='YlOrRd',  # Yellow-Orange-Red colormap
                fmt='.2f',
                annot=True,
                square=True,
                cbar=True,
                vmin=0,
                vmax=1,
                annot_kws={'size': 8})
    
    # Customize first subplot
    ax1.set_xlabel('k0 k1 k2 k3 k4 k5 k6 k7 k8 k9')
    ax1.set_ylabel('q0 q1 q2 q3 q4 q5 q6 q7 q8 q9')
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    
    # Customize second subplot
    ax2.set_xlabel('k0 k1 k2 k3 k4 k5 k6 k7 k8 k9')
    ax2.set_ylabel('q0 q1 q2 q3 q4 q5 q6 q7 q8 q9')
    ax2.xaxis.set_ticks_position('top')
    ax2.xaxis.set_label_position('top')
    
    # Create example data matching your image
    # You would replace these with your actual attention matrices
    plt.tight_layout()
    return fig
def plot_attention_scores(attention_scores, title="Attention Scores Heatmap", cmap='RdPu'):
    """
    Plot attention scores as a heatmap.
    
    Args:
        attention_scores (torch.Tensor or numpy.ndarray): Matrix of attention scores
        title (str): Title for the plot
        cmap (str): Colormap to use for the heatmap
    """
    # Convert to numpy if tensor
    if torch.is_tensor(attention_scores):
        attention_scores = attention_scores.detach().cpu().numpy()
    
    # Create figure and axes
    plt.figure(figsize=(12, 5))
    
    # Create two subplots
    plt.subplot(1, 2, 1)
    
    # Plot first heatmap with larger values
    sns.heatmap(attention_scores, 
                annot=True, 
                cmap=cmap,
                fmt='.2f',
                square=True,
                cbar=True,
                #x axis plots Keys, y axis plots query 
                xticklabels=[f'k{i}' for i in range(attention_scores.shape[1])],
                yticklabels=[f'q{i}' for i in range(attention_scores.shape[0])])
    
    plt.title(f'{title} (Standard Scale)')
    
    # Plot second heatmap with smaller values
    plt.subplot(1, 2, 2)
    
    # Create a mask for values below threshold
    small_values = np.where(attention_scores < 0.1, attention_scores, 0)
    
    sns.heatmap(small_values,
                annot=True,
                cmap='YlOrRd',
                fmt='.2f',
                square=True,
                cbar=True,
                xticklabels=[f'k{i}' for i in range(attention_scores.shape[1])],
                yticklabels=[f'q{i}' for i in range(attention_scores.shape[0])])
    
    plt.title(f'{title} (Small Values Focus)')
    
    # Adjust layout
    plt.tight_layout()
    return plt.gcf()

def get_attention_scores(model, data):
    """
    Extract attention scores from a GATv2 or GAT model for a single graph.
    
    Args:
        model (torch.nn.Module): The GAT or GATv2 model
        data (torch_geometric.data.Data): A single graph
        
    Returns:
        torch.Tensor: Attention scores from the first attention layer
    """
    model.eval()
    with torch.no_grad():
        # Forward pass through first attention layer only
        if isinstance(model, GATv2):
            x = model.conv1(data.x, data.edge_index)
            # Get attention scores from the first attention layer
            attention = model.conv1.alpha
        else:  # GAT model
            x, (edge_index, attention) = model.conv1(data.x, data.edge_index, return_attention_weights = True)
            # Get attention scores from the first attention layer
        
        return attention

def visualize_attentions(model1, model2, test_loader, device):
    """
    Visualize attention scores for both models on a sample graph.
    
    Args:
        model1 (GATv2): The GATv2 model
        model2 (GAT): The GAT model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run the models on
    """
    # Get a single batch
    sample_data = next(iter(test_loader)).to(device)
    
    # Get attention scores
    attention_scores_gatv2 = get_attention_scores(model1, sample_data)
    attention_scores_gat = get_attention_scores(model2, sample_data)
    
    # Plot attention scores
    plot_attention_comparison(attention_scores_gatv2.cpu().numpy(), attention_scores_gat.cpu().numpy())
