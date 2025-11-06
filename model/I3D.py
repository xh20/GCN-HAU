import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class Unit3D(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, use_batch_norm=True,
                 activation_fn=nn.ReLU(), use_bias=False):
        super(Unit3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=output_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=use_bias)
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm3d(output_channels)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.conv3d(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class InceptionBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels_1x1, out_channels_3x3_reduce, out_channels_3x3, out_channels_5x5_reduce, out_channels_5x5, out_channels_pool_proj):
        super(InceptionBlock3D, self).__init__()
        # 1x1 conv branch
        self.branch1 = Unit3D(in_channels, out_channels_1x1, kernel_size=1)

        # 1x1 -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            Unit3D(in_channels, out_channels_3x3_reduce, kernel_size=1),
            Unit3D(out_channels_3x3_reduce, out_channels_3x3, kernel_size=3, padding=1)
        )

        # 1x1 -> 5x5 conv branch
        self.branch3 = nn.Sequential(
            Unit3D(in_channels, out_channels_5x5_reduce, kernel_size=1),
            Unit3D(out_channels_5x5_reduce, out_channels_5x5, kernel_size=5, padding=2)
        )

        # Pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            Unit3D(in_channels, out_channels_pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class I3D_Skeleton(nn.Module):
    def __init__(self, num_classes=60, input_channels=3, num_joints=25):
        super(I3D_Skeleton, self).__init__()

        # Initial Conv + MaxPool layers
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0))
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        self.conv2 = nn.Conv3d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv3d(64, 192, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        # Inception blocks
        self.inception3a = InceptionBlock3D(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock3D(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        self.inception4a = InceptionBlock3D(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock3D(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock3D(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock3D(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock3D(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        self.inception5a = InceptionBlock3D(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock3D(832, 384, 192, 384, 48, 128, 128)

        # Final layers
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Forward pass through layers
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        # Final pooling, dropout, and classification
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class I3D_small(nn.Module):
    def __init__(self, in_channels=3, out_channels=512):
        super(I3D_small, self).__init__()
        self.bn3d = nn.BatchNorm3d(in_channels)

        # Initial Conv + MaxPool layers
        self.conv1 = Unit3D(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 3, 3), padding=(1, 3, 3))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv2 = Unit3D(64, 192, kernel_size=1)
        self.conv3 = Unit3D(192, 192, kernel_size=(3, 7, 7), stride=(1, 3, 3), padding=(1, 3, 3))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Inception blocks
        self.inception3a = InceptionBlock3D(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock3D(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.inception4a = InceptionBlock3D(480, 192, 96, 208,
                                            16, 48, 64)
        self.inception4b = InceptionBlock3D(512, 160, 112, 224,
                                            24, 64, 64)
        self.inception4c = InceptionBlock3D(512, 128, 128, 256,
                                            24, 64, 64)
        self.inception4d = InceptionBlock3D(512, 112, 144, 288,
                                            32, 64, 64)
        self.inception4e = InceptionBlock3D(528, 256, 160, 320,
                                            32, 128, 128)
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.inception5a = InceptionBlock3D(832, 256, 160, 320,
                                            32, 128, 128)
        self.inception5b = InceptionBlock3D(832, 384, 192, 384,
                                            48, 128, 128)
        # Final layers
        self.conv_out = Unit3D(1024, out_channels, kernel_size=(1, 1, 1),
                               use_batch_norm=True, use_bias=True)
        # self.linear = nn.Linear(1024, out_channels)
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        # Apply weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Use Kaiming initialization (He initialization) for Conv3D
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Use Xavier (Glorot) initialization for fully connected layers
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                # Initialize BatchNorm3d layers (if any are used)
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, image):
        x = self.bn3d(image)
        # Forward pass through layers
        x1 = self.conv1(x)
        x2 = self.pool1(x1)

        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x = self.pool2(x2)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        # Final layer
        x = self.conv_out(x)
        # x = self.avgpool2d(x)

        return x


class CrossModalAttention(nn.Module):
    def __init__(self, motion_dim, img_dim, rel_reduction=8):
        super(CrossModalAttention, self).__init__()
        rel_dim = min(motion_dim, img_dim) // rel_reduction
        self.motion_query = nn.Conv2d(motion_dim, rel_dim, kernel_size=(1, 1))
        self.img_query = nn.Conv3d(img_dim, rel_dim, kernel_size=(1, 1, 1))
        self.img_value = nn.Conv3d(img_dim, img_dim, kernel_size=(1, 1, 1))
        self.att = nn.Conv2d(rel_dim, img_dim, kernel_size=(1, 1))
        self.tanh = nn.Tanh()

    def forward(self, image, motion):
        # x1_: image/flow, x2_: motion
        N, C1, T1, H, W = image.shape
        _, C2, T2, V = motion.shape
        Q = self.img_query(image).view(N, -1, T1, H*W).mean(-1)
        K = self.motion_query(motion)
        K = F.adaptive_avg_pool2d(K, (T1, 1))
        V = self.img_value(image).view(N, -1, T1, H*W)

        scores_t = self.tanh(K - Q.unsqueeze(-2))  # N, c, T2, T1
        scores_t = self.att(scores_t) / T1 + torch.eye(T1, dtype=scores_t.dtype, device=scores_t.device).unsqueeze(0).unsqueeze(0)
        y = torch.einsum('ncij,ncjv->nciv', scores_t, V).view(N, -1, T1, H, W)  # N x C x T1 x 1

        return y


class I3D_early_fusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=512):
        super(I3D_early_fusion, self).__init__()
        self.bn3d = nn.BatchNorm3d(in_channels)

        # Initial Conv + MaxPool layers
        self.conv1 = Unit3D(in_channels, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3))
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.cross_att1 = CrossModalAttention(64, 64)

        self.conv2 = Unit3D(64, 192, kernel_size=1)
        self.conv3 = Unit3D(192, 192, kernel_size=(3, 7, 7), padding=(1, 3, 3))
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))

        # Inception blocks
        self.inception3a = InceptionBlock3D(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock3D(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.cross_att2 = CrossModalAttention(128, 192)

        self.inception4a = InceptionBlock3D(480, 192, 96, 208,
                                            16, 48, 64)
        self.inception4b = InceptionBlock3D(512, 160, 112, 224,
                                            24, 64, 64)
        self.inception4c = InceptionBlock3D(512, 128, 128, 256,
                                            24, 64, 64)
        self.inception4d = InceptionBlock3D(512, 112, 144, 288,
                                            32, 64, 64)
        self.inception4e = InceptionBlock3D(528, 256, 160, 320,
                                            32, 128, 128)
        self.pool4 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.cross_att3 = CrossModalAttention(256, 480)

        self.inception5a = InceptionBlock3D(832, 256, 160, 320,
                                            32, 128, 128)
        self.inception5b = InceptionBlock3D(832, 384, 192, 384,
                                            48, 128, 128)
        # Final layers
        self.conv_out = Unit3D(1024, out_channels, kernel_size=(1, 1, 1), activation_fn=None,
                               use_batch_norm=True, use_bias=True)
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        # Apply weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Use Kaiming initialization (He initialization) for Conv3D
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Use Xavier (Glorot) initialization for fully connected layers
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                # Initialize BatchNorm3d layers (if any are used)
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, image, c1, c2, c3):
        x = self.bn3d(image)
        # Forward pass through layers
        x1 = self.conv1(x)
        x1 = self.cross_att1(x1, c1)
        x2 = self.pool1(x1)

        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.cross_att2(x2, c2)
        x3 = self.pool2(x2)

        x3 = self.inception3a(x3)
        x3 = self.inception3b(x3)
        x3 = self.cross_att3(x3, c3)
        x4 = self.pool3(x3)

        x4 = self.inception4a(x4)
        x4 = self.inception4b(x4)
        x4 = self.inception4c(x4)
        x4 = self.inception4d(x4)
        x4 = self.inception4e(x4)
        x5 = self.pool4(x4)

        x5 = self.inception5a(x5)
        y = self.inception5b(x5)

        # Final layer
        # x = self.conv_out(x)
        # x = self.avgpool2d(x)
        return y


# Example usage:
if __name__ == '__main__':
    # model = I3D_Skeleton(num_classes=60, input_channels=3, num_joints=25)  # num_joints depends on dataset (25 joints for Kinect)
    model = I3D_small(input_channels=3, out_channels=512)  # num_joints depends on dataset (25 joints for Kinect)
    inputs = torch.randn(4, 3, 8, 480, 480)  # (batch_size, channels, frames, joints)
    outputs = model(inputs)
    print(outputs.shape)  # Expected output shape: (batch_size, num_classes)
