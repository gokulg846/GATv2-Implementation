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

'''
Implementing GATv2 and GAT

Defining custom GATv2 Conv Layer based on the paper: https://arxiv.org/abs/2105.14491
Utilizing inbuilt GAT model in PyTorch Geometric
Comparing Mean Absolute Error (MAE) and training losses between GAT and GATv2
Both models are trained on QM9 dataset with the same parameters
'''
class GATv2Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=4, concat=True, negative_slope=0.2, dropout=0.0, share_weights=False):
        super(GATv2Conv, self).__init__(aggr='add', node_dim=0)

        '''
        Args:
            in_channels: Number of input features
            out_channels: Number of output features
            heads: Number of attention heads
            concat - If True, the output of the different heads is concatenated
                     If False, the output of the different heads is averaged
            negative_slope: LeakyReLU angle of negative slope
            dropout: Dropout probability
            share_weights: If True, the same linear transformation will be applied to both source and target nodes
        
        '''

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.share_weights = share_weights

        self.lin_l = Linear(in_channels, heads * out_channels)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels)

        self.attention = Parameter(torch.Tensor(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin_l.weight)
        torch.nn.init.xavier_uniform_(self.lin_r.weight)
        torch.nn.init.xavier_uniform_(self.attention)

    def forward(self, x, edge_index):
      '''
      Inputs: 
      x - Input features per node
      edge_index - Sparse matrix representing the graph
      Returns:
      torch.Tensor: Updated node features after applying attentions
      '''
        if isinstance(x, torch.Tensor):
            assert x.dim() == 2
            Wh_l = self.lin_l(x).view(-1, self.heads, self.out_channels)
            if self.share_weights:
                Wh_r = Wh_l
            else:
                Wh_r = self.lin_r(x).view(-1, self.heads, self.out_channels)
        else:
            Wh_l, Wh_r = x
            assert x[0].dim() == 2
            Wh_l = self.lin_l(Wh_l).view(-1, self.heads, self.out_channels)
            if Wh_r is not None:
                Wh_r = self.lin_r(Wh_r).view(-1, self.heads, self.out_channels)

        return self.propagate(edge_index, x=(Wh_l, Wh_r))

    def message(self, x_j, x_i, index, size_i):

      '''
      Args:
            x_j: Source node features
            x_i: Target node features
            index: Index tensor
            size_i: Size of target nodes
      
      Alpha - Attention scores
      Returns:
            torch.Tensor: Messages weighted by attention coefficients

      '''

        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.attention).sum(dim=-1)
        alpha = softmax(alpha, index, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def update(self, out):

      '''

      Inputs: 
      out - Output features per node of the central node
      Computes: 
      If concat is True, the output of the different heads is concatenated
      If concat is False, the output of the different heads is averaged
      Returns:
      Averaged/ Concatenated Output features per node of the central node

      '''

        if self.concat:
            return out.view(-1, self.heads * self.out_channels)
        else:
            return out.mean(dim=1)

class GATv2(torch.nn.Module):
    '''
    GATv2 model - Implements 3 convolutional layers followed by global pooling and a single final linear layer
    '''
    def __init__(self, in_channels, hidden_channels, out_channels):
        '''
        Args:
            in_channels: Number of input features
            hidden_channels: Number of hidden features
            out_channels: Number of output features

          
        '''
        super(GATv2, self).__init__()

        # First GATv2 layer
        self.conv1 = GATv2Conv(in_channels, hidden_channels, concat=True)

        # Second GATv2 layer
        self.conv2 = GATv2Conv(hidden_channels * self.conv1.heads, hidden_channels, concat=True)

        self.conv3 = GATv2Conv(hidden_channels * self.conv2.heads, hidden_channels, concat=False)

        # Single linear layer for prediction
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
      '''
      Args:
            x - Input features per node
            edge_index - Sparse matrix representing the graph
            batch - Batch vector
      Returns:
            torch.Tensor: Predicted node features
      '''
        # First GATv2 layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Second GATv2 layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Third GATv2 layer - Mean pooling applied after this layer
        x = self.conv3(x, edge_index)
        # Global pooling
        x = global_mean_pool(x, batch)

        # Final prediction with single linear layer
        x = self.lin(x)

        return x

class GAT(torch.nn.Module):\
  '''
  Using inbuilt GAT model in PyTorch Geometric
  GAT model - Implements 3 convolutional layers followed by global pooling and a single final linear layer
  '''
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()

        # First GATv2 layer
        self.conv1 = GATConv(in_channels, hidden_channels, concat=True)

        # Second GATv2 layer
        self.conv2 = GATConv(hidden_channels * self.conv1.heads, hidden_channels, concat=True)

        self.conv3 = GATConv(hidden_channels * self.conv2.heads, hidden_channels, concat=False)

        # Single linear layer for prediction
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # First GATv2 layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Second GATv2 layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Third GATv2 layer
        x = self.conv3(x, edge_index)
        # Global pooling
        x = global_mean_pool(x, batch)

        # Final prediction with single linear layer
        x = self.lin(x)

        return x


def train(model, loader, optimizer, device):
  '''
  Training function for one epoch.
    
    Args:
        model (GATv2): The GATv2 model
        loader (DataLoader): DataLoader for training data
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates
        device (torch.device): Device to run the model on
        
    Returns:
        float: Average training loss per graph for this epoch
  '''
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.mse_loss(out, data.y[:, 0].unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


def test(model, loader, device):
  '''
  Testing function to evaluate model performance.
    
    Args:
        model (GATv2): The GATv2 model
        loader (DataLoader): DataLoader for test data
        device (torch.device): Device to run the model on
        
    Returns:
        float: Average Mean Absolute Error (MAE) across all graphs in the dataset
  '''
    model.eval()
    total_mae = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            mae = (out - data.y[:, 0].unsqueeze(1)).abs().sum()
            total_mae += mae.item()

    return total_mae / len(loader.dataset)


def plot_training_curves(loss1, loss2, train: bool):
  '''
  Plot and save training loss and validation MAE curves
  Args:
    loss1, loss2: Loss values for GATv2 and GAT models respectively
    train: If true, the losses are training losses, else validation MAEs

  Returns:
    matplotlib.figure.Figure: Figure object containing the plot

  '''
    if train:
      plt.figure(figsize=(8, 6))
      plt.plot(loss1, label='Training Loss GATv2')
      plt.plot(loss2, label='Training Loss GAT')
      plt.xlabel('Epoch')
      plt.ylabel('Value')
      plt.title('Training Loss Curves GATv2 vs GAT')
      plt.legend()
      plt.tight_layout()
      return plt.gcf()
    else:
      plt.figure(figsize=(8, 6))
      plt.plot(loss1, label='Validation MAE GATv2')
      plt.plot(loss2, label='Validation MAE GAT')
      plt.xlabel('Epoch')
      plt.ylabel('Value')
      plt.title('Validation MAE Curves GATv2 vs GAT')
      plt.legend()
      plt.tight_layout()
      return plt.gcf()


def plot_true_vs_pred(model, loader, device):
  '''
  Create and save scatter plot of predicted vs true values.
    
    Args:
        model (GATv2): The GATv2 model
        loader (DataLoader): DataLoader for test data
        device (torch.device): Device to run the model on
        
    Returns:
        matplotlib.figure.Figure: Figure object containing the plot

  '''
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            y_true.append(data.y[:, 0].cpu())
            y_pred.append(out.cpu())
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs. True Values')
    plt.tight_layout()
    return plt.gcf()
'''
Uncomment code to get model performance table as a dataframe
def get_model_performance_table(train_loss, val_mae, test_mae):
    metrics = pd.DataFrame({
        'Metric': ['Training Loss', 'Validation MAE', 'Test MAE'],
        'Value': [train_loss, val_mae, test_mae]
    })
    return metrics
'''
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load QM9 dataset
    dataset = QM9(root='data/QM9')
    
    # Split dataset
    train_dataset = dataset[:9000]
    val_dataset = dataset[9000:11000]
    test_dataset = dataset[11000:12000]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize GATv2 model
    model = GATv2(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=1  # Predicting single target property
    ).to(device)

    #Initialize GAT model
    model2 = GAT(
        in_channels=dataset.num_features,
        hidden_channels=64,
        out_channels=1  # Predicting single target property
    ).to(device)
    
    # Initialize separate optimizers
    optimizer1 = torch.optim.Adam(model.parameters(), lr=0.0015)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0015)
    train_losses1 = [] #For GATv2
    val_maes1 = [] #For GATv2
    train_losses2 = [] #For GAT
    val_maes2 = [] #For GAT
    best_val_mae1 = float('inf') #For GATv2
    best_val_mae2 = float('inf') #For GAT

    print("Running GATv2 model")
    for epoch in range(100):

        train_loss1 = train(model, train_loader, optimizer1, device)
        val_mae1 = test(model, val_loader, device)
        
        # Save best model
        if val_mae1 < best_val_mae1:
            best_val_mae = val_mae1
            torch.save(model.state_dict(), 'best_model.pth')
        
        train_losses1.append(train_loss1)
        
        
        if epoch % 10 == 0:
          print(f'Epoch: {epoch:03d}, Train Loss: {train_loss1:.4f}, Val MAE: {val_mae1:.4f}')
          val_maes1.append(val_mae1)

    print("Running GAT model")
    for epoch in range(100):
        train_loss2 = train(model2, train_loader, optimizer2, device)
        val_mae2 = test(model2, val_loader, device)
        
        # Save best model
        if val_mae2 < best_val_mae2:
            best_val_mae2 = val_mae2
            torch.save(model2.state_dict(), 'best_model2.pth')
        
        train_losses2.append(train_loss2)
        
        
        if epoch % 10 == 0:
          print(f'Epoch: {epoch:03d}, Train Loss: {train_loss2:.4f}, Val MAE: {val_mae2:.4f}')
          val_maes2.append(val_mae2)
    
    # Load best GATv2 model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    test_mae1 = test(model, test_loader, device)
    print(f'Test MAE GATv2 Model: {test_mae1:.4f}')

    # Load best GAT model and evaluate on test set
    model2.load_state_dict(torch.load('best_model2.pth'))
    test_mae2 = test(model2, test_loader, device)
    print(f'Test MAE GAT Model: {test_mae2:.4f}')

    # Plot training and validation curves
    plot_training_curves(train_losses1, train_losses2, train = True)
    plot_training_curves(val_maes1, val_maes2, train = False)
    
    # Plot predicted vs. true values
    plot_true_vs_pred(model, test_loader, device)
    plot_true_vs_pred(model2, test_loader, device)

    
    # Uncomment to execute Get model performance table
    #model_performance = get_model_performance_table(train_loss, val_mae, test_mae)
    #print(model_performance)

if __name__ == '__main__':
    main()
