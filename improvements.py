import torch
import torch.nn as nn
import torch.nn.functional as F


def get_blur_filter(sigma, kernel_size, device):
    # Get the 2D Gaussian blue kernel
    # Adapted to torch from https://www.geeksforgeeks.org/how-to-generate-2-d-gaussian-array-using-numpy/

    # Create horizontal and vertical kernel
    x, y = torch.meshgrid( torch.linspace(-1, 1, kernel_size) , torch.linspace(-1, 1, kernel_size), indexing="xy" )

    x = x.to(device)
    y = y.to(device)

    # Create the filter using the 2D Gaussian equation 
    filter = torch.exp(-((x ** 2) + (y ** 2)) / (2 * (sigma ** 2)))

    # Normalize
    filter = F.normalize(filter)

    # To use kernel in conv2d function kernel should be shape  ( out_channels=3 , in_channels=3 , kernel_size , kernel_size) 
    filter = torch.stack((filter, filter, filter), dim=0) # input channels
    filter = torch.stack((filter, filter, filter), dim=0) # output channels

    return filter


class BlurLayer(nn.Module):
    def __init__(self, in_channels=3, kernel_size=5, sqrt_sigma_initial=1.0):
        super(BlurLayer, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        # Train for the sqrt of the sigma to keep sigma positie when calculating Gaussian
        self.sqrt_sigma = nn.Parameter( torch.tensor(sqrt_sigma_initial, dtype=torch.float32) )

    def forward(self, x):
        # Applying blurring to x using a learned level of blurring

        # Get actual sigma
        sigma = self.sqrt_sigma ** 2

        # Get gaussian filter from the learned sigma
        filter = get_blur_filter(sigma, self.kernel_size, x.device)

        return F.conv2d(x, weight=filter, padding='same')


class ImprovedMrCNNStream(nn.Module):
    def __init__(self, blur_kernel_size, blur_sqrt_sigma_initial):
        super(ImprovedMrCNNStream, self).__init__()
        
        self.blur1 = BlurLayer(in_channels=3, kernel_size=blur_kernel_size, sqrt_sigma_initial=blur_sqrt_sigma_initial)
    
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=160, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=160, out_channels=288, kernel_size=3, stride=1, padding=0)

        self.dropout1 = nn.Dropout(p=0.5)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(in_features=(288 * 3 * 3), out_features=512)

        self.dropout2 = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Blur layer
        x = self.blur1(x)

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

        self.stream1 = ImprovedMrCNNStream(blur_kernel_size = 7, blur_sqrt_sigma_initial = 0.5)
        self.stream2 = ImprovedMrCNNStream(blur_kernel_size = 5, blur_sqrt_sigma_initial = 0.75)
        self.stream3 = ImprovedMrCNNStream(blur_kernel_size = 3, blur_sqrt_sigma_initial = 1.0)

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