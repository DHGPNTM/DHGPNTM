import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
import torch.nn.init as init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GPNConv(torch.nn.Module):
    def __init__(self, nn):
        super(GPNConv, self).__init__()
        self.nn = nn

    def forward(self, x, edge_index, weight):
        row, col = edge_index

        deg = scatter_add(weight, col, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        weight = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]

        weight_new = weight.unsqueeze(dim=-1).expand(-1, x.size(1)) * x[row]

        out = scatter_add(weight_new, col, dim=0, dim_size=x.size(0))

        out = x + out
        out = self.nn(out)


        # 聚合邻居节点的特征 row -> col ???????????????
        # out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))
        # out = x + out
        # out = self.nn(out)
        return out

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers


        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            self.init_weights()


    def init_weights(self):
        for linear in self.linears:
            init.xavier_normal_(linear.weight)

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class GPN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim):
        super(GPN, self).__init__()
        self.gpn_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.num_layers = num_layers
        for layer in range(num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            gpn_conv = GPNConv(mlp)
            self.gpn_layers.append(gpn_conv)

            batch_norm = nn.BatchNorm1d(hidden_dim)
            self.batch_norms.append(batch_norm)

        self.linears_prediction = torch.nn.ModuleList()
        middle_dim = 128
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(input_dim, middle_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, middle_dim))

        self.output_layer = nn.Linear(middle_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        for linear in self.linears_prediction:
            init.xavier_normal_(linear.weight)
        init.xavier_normal_(self.output_layer.weight)

    def forward(self, x, A, weight):
        h = x
        hidden_rep = [h]
        for layer in range(self.num_layers - 1):
            h = self.gpn_layers[layer](h, A, weight)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            hidden_rep.append(h)

        output_h = 0
        for layer, h in enumerate(hidden_rep):
            if not(layer == 0 or layer == len(hidden_rep)-1):
                continue

            output_h += self.linears_prediction[layer](h)

        output_h = F.relu(output_h).to(device)
        outputs = self.output_layer(output_h)

        return outputs


if __name__ == '__main__':
    from torch_geometric.datasets import TUDataset
    from torch_geometric.data import DataLoader

    dataset = TUDataset(root='./tmp/ENZYMES', name='ENZYMES')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    data = None
    for d in loader:
        data = d
        break
    gpn = GPN(2, 2, 3, 8, 5)
    outputs = gpn(data.x, data.edge_index, data.batch)
    print(outputs.shape)

