import torch
import torch.nn as nn
import torchvision


class DenseNet(nn.Module):
    def __init__(self, fixed_extractor=True):
        super(DenseNet, self).__init__()
        # Load pretrained original densenet
        original_model = torchvision.models.densenet121(weights="DEFAULT")

        # Freeze weights
        if fixed_extractor:
            for param in original_model.parameters():
                param.requires_grad = False

        # Adapt backend
        new_model = list(original_model.children())[:-1]
        new_model.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.features = torch.nn.Sequential(*new_model)

        self.classifier = None
        self.top_layer = None

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class MLP(nn.Module):
    def __init__(self, n_input, n_hid=64, n_out=2):
        super().__init__()

        self.n_input = n_input
        self.linear1 = nn.Linear(n_input, n_hid)
        self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(n_hid, n_hid)
        self.output = nn.Linear(n_hid, n_out)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        X = X.to(torch.float32)
        X = self.nonlin(self.linear1(X))
        # X = self.linear1(X)
        # X = self.dropout(X)
        # X = self.nonlin(self.linear2(X))
        X = self.softmax(self.output(X))
        return X
