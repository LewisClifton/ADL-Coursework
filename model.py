import torch
import torch.nn as nn
import torch.nn.functional as F

# The following model is a recreation of the MrCNN (Multi-resolution Convolutional Neural Network) described
# by Liu, N., Han, J., Zhang, D., Wen, S., & Liu, T. (2015). Predicting eye fixations using convolutional neural networks. 
# In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 362-370). Found here:
# https://openaccess.thecvf.com/content_cvpr_2015/papers/Liu_Predicting_Eye_Fixations_2015_CVPR_paper.pdf


# The model structure is justified in the comments which quote the reference paper's description of the original Mr-CNN

class MrCNNStream(nn.Module):
    def __init__(self):
        super(MrCNNStream, self).__init__()
        
        # "96 filters with size 7×7 in the first C layer"
        # "160 and 288 filters with size 3×3 respectively in the second and the third C layer"
        # "Stride to 1 and perform valid convolution operations, disregarding the map borders" (implies padding=0)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=160, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=160, out_channels=288, kernel_size=3, stride=1, padding=0)

        # "Dropout was used with the corruption probability of 0.5 in the third C layer"
        self.dropout1 = nn.Dropout(p=0.5)

        # "Use 2×2 pooling windows in all P layers"
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # "512 neurons in all FC layers"
        self.fc = nn.Linear(in_features=(288 * 3 * 3), out_features=512)

        # "Dropout was used with the corruption probability of 0.5 in ... and the subsequent two FC layers
        self.dropout2 = nn.Dropout(p=0.5)

        # "Rectified Linear Unit (ReLU) in all C layers and FC layer"
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
        
        # FC layer at end of stream
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.dropout2(x)
        out = self.relu(x)

        # Return stream output
        return out

class MrCNN(nn.Module):
    def __init__(self):
        super(MrCNN, self).__init__()

        # "Mr-CNN starts from three streams in lower layer"
        self.stream1 = MrCNNStream()
        self.stream2 = MrCNNStream()
        self.stream3 = MrCNNStream()
        
        # "The three streams are fused using another FC layer"
        # "512 neurons in all FC layers"
        self.fc = nn.Linear(in_features=(512 * 3), out_features=512)

        # "Dropout was used with the corruption probability of 0.5 in ... and the subsequent two FC layers
        self.dropout = nn.Dropout(p=0.5)

        # "Followed by one logistic regression layer at the end to perform classification"
        # "512 neurons in all FC layers"
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