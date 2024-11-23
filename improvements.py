import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianPriorMap(nn.Module):
    def __init__(self, N=16):
        super(GaussianPriorMap, self).__init__()

        self.N = N

        self.mean_x = nn.Parameter(torch.randn(N))
        self.mean_y = nn.Parameter(torch.randn(N))
        self.std_x = nn.Parameter(torch.randn(N)) 
        self.std_y = nn.Parameter(torch.randn(N))
    
    @staticmethod
    def gaussian_f(x, y, mean_x, mean_y, std_x, std_y):
        return (1 / (2 * np.pi * std_x * std_y)) * torch.exp(-((x - mean_x) ** 2 / (2 * std_x ** 2) + (y - mean_y) ** 2 / (2 * std_y ** 2)))

    def forward(self, map_width, map_height):

        grid_x, grid_y = torch.meshgrid(torch.arange(0, map_width), torch.arange(0, map_height))
        grid_x = grid_x.float()
        grid_y = grid_y.float()

        for i in range(self.N):
            gaussian_map = GaussianPriorMap.gaussian_f(grid_x, grid_y, self.mu_x[i], self.mu_y[i], self.sigma_x[i], self.sigma_y[i])
            gaussian_maps.append(gaussian_map)

        gaussian_maps = [GaussianPriorMap.gaussian_f(grid_x, grid_y, self.mu_x[i], self.mu_y[i], self.sigma_x[i], self.sigma_y[i]) for _ in range(self.N)]

        gaussian_maps = torch.stack(gaussian_maps, dim=0).unsqueeze(0)

        return gaussian_map

class MrCNNStream(nn.Module):
    def __init__(self):
        super(MrCNNStream, self).__init__()
        
    
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=160, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=160, out_channels=288, kernel_size=3, stride=1, padding=0)

        self.dropout1 = nn.Dropout(p=0.5)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gaussian_priors = GaussianPriorMap(N=10)

        self.conv4 = nn.Conv2d(in_channels=288 + 10, out_channels=512, kernel_size=3, stride=1, padding=0)

        self.fc = nn.Linear(in_features=(512 * 3 * 3), out_features=512)

        self.dropout2 = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):
        # First layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Second layer
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Third layer
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.pool(x)

        # Gaussian maps
        gaussian_maps = self.gaussian_priors(x.shape[3], x.shape[4])
        x = torch.cat([x, gaussian_maps], dim=1)
        x = self.conv4(x)

        
        # FC layer at end of stream
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.dropout2(x)
        out = self.relu(x)

        # Return stream output
        return out

class ImprovedMrCNN(nn.Module):
    def __init__(self):
        super(ImprovedMrCNN, self).__init__()

        self.stream1 = MrCNNStream()
        self.stream2 = MrCNNStream()
        self.stream3 = MrCNNStream()

        self.fc = nn.Linear(in_features=(512 * 3), out_features=512)

        self.dropout = nn.Dropout(p=0.5)

        self.output = nn.Linear(in_features=512, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3):

        # Run the input through the three streams
        stream_out1 = self.stream1(x1)
        stream_out2 = self.stream2(x2)
        stream_out3 = self.stream3(x3)

        # Fuse the stream outputs using the FC layer
        streams_out = torch.cat((stream_out1, stream_out2, stream_out3), dim=1)
        fused = self.fc(streams_out)
        fused = F.relu(fused)
        fused = self.dropout(fused)
        
        # Get the output from the logistic regression layer to perform classification
        logit = self.output(fused)
        out = self.sigmoid(logit)
        
        # Return MrCNN output
        return out