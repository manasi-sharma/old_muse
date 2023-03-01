import torch
import torch.nn as nn

from attrdict import AttrDict
from attrdict.utils import get_with_default, get_or_instantiate_cls


class Conv3DFeatureModel(torch.nn.Module):
    """
    - A 3D CNN with 11 layers.
    - Kernel size is kept 3 for all three dimensions - (time, H, W)
      except the first layer has kernel size of (3, 5, 5)
    - Time dimension is preserved with `padding=1` and `stride=1`, and is
      averaged at the end
    Arguments:
    - Input: a (batch_size, 3, sequence_length, W, H) tensor
    - Returns: a (batch_size, 512) sized tensor
    """

    def __init__(self, params: AttrDict):
        super(Conv3DFeatureModel, self).__init__()

        feature_size = get_with_default(params, "feature_size", 512, map_fn=int)

        self.block1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block4 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, feature_size, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_size),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
        )

        self.block5 = nn.Sequential(
            nn.Conv3d(feature_size, feature_size, kernel_size=(3, 3, 3), stride=1, dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_size),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_size, feature_size, kernel_size=(3, 3, 3), stride=(1, 2, 2), dilation=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # get convolution column features

        x = self.block1(x)
        # print(x.size())
        x = self.block2(x)
        # print(x.size())
        x = self.block3(x)
        # print(x.size())
        x = self.block4(x)
        # print(x.size())
        x = self.block5(x)
        # print(x.size())

        # averaging features in time dimension
        x = x.mean(-1).mean(-1).mean(-1)

        return x


class MultiColumn(nn.Module):

    def __init__(self, num_classes, conv_column, column_units,
                 clf_layers=None):
        """
        - Example multi-column network
        - Useful when a video sample is too long and has to be split into
          multiple clips
        - Processes 3D-CNN on each clip and averages resulting features across
          clips before passing it to classification(FC) layer
        Args:
        - Input: Takes in a list of tensors each of size
                 (batch_size, 3, sequence_length, W, H)
        - Returns: logits of size (batch size, num_classes)
        """
        super(MultiColumn, self).__init__()
        self.num_classes = num_classes
        self.column_units = column_units
        if isinstance(conv_column, nn.Module):
            self.conv_column = conv_column
        else:
            self.conv_column = get_or_instantiate_cls(conv_column, '', nn.Module)
        self.clf_layers = clf_layers

        if not self.clf_layers:
            self.clf_layers = nn.Sequential(
                              nn.Linear(column_units, self.num_classes)
                             )

    def forward(self, inputs, get_features=False):
        outputs = []
        num_cols = len(inputs)
        for idx in range(num_cols):
            x = inputs[idx]
            x1 = self.conv_column(x)
            outputs.append(x1)

        outputs = torch.stack(outputs).permute(1, 0, 2)
        outputs = torch.squeeze(torch.sum(outputs, 1), 1)
        avg_output = outputs / float(num_cols)
        outputs = self.clf_layers(avg_output)
        if get_features:
            return outputs, avg_output
        else:
            return outputs


if __name__ == "__main__":
    num_classes = 174
    input_tensor = torch.autograd.Variable(torch.rand(5, 3, 72, 84, 84))
    model = Conv3DFeatureModel(AttrDict(feature_size=512))

    output = model(input_tensor)
    assert list(output.size()) == [5, 512]

    input_tensor = [torch.autograd.Variable(torch.rand(1, 3, 72, 84, 84))]
    # two ways to instantiate
    model2 = MultiColumn(174, AttrDict(cls=Conv3DFeatureModel, params=AttrDict(feature_size=512)), 512)
    model3 = MultiColumn(174, model, 512)
    output2 = model2(input_tensor)
    assert list(output2.size()) == [1, num_classes]
    output3 = model3(input_tensor)
    assert list(output3.size()) == [1, num_classes]
