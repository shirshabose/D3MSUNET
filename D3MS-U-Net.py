
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """
 
    def __init__(self, scale_factor, mode="bilinear"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
 
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True, recompute_scale_factor=True)
        return x    
        
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
 
 
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
 
 
class BasicBlock(nn.Module):
    expansion: int = 1
 
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = True,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.conv1 = conv3x3(inplanes, planes, stride,dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,stride=1,dilation=dilation)
        self.bn2 = norm_layer(planes)
        if downsample == True:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif isinstance(downsample, nn.Module):
            self.downsample = downsample
        else:
            self.downsample = None
        self.stride = stride
 
    def forward(self, x: Tensor) -> Tensor:
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            identity = self.downsample(identity)
 
        out += identity
        out = self.relu(out)
 
        return out
 
 
class BasicBlock_2(nn.Module):
    expansion: int = 1
 
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock_2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x: Tensor) -> Tensor:
 
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        
        out = self.relu(out)
 
        return out
 
 
class ResNet(nn.Module):
 
    def __init__(
        self,
        initial_channel: int, 
        block: Type[Union[BasicBlock, BasicBlock_2]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
 
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:

            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(initial_channel, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
 
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1,
                                       dilate=False,dilation=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=False,dilation=4)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=False,dilation=8)
 
        self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool3=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool4=nn.MaxPool2d(kernel_size=2,stride=2)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
 
    def _make_layer(self, block: Type[Union[BasicBlock, BasicBlock_2]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False,dilation: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,stride=1,groups=self.groups,
                                base_width=self.base_width,dilation=dilation,
                                norm_layer=norm_layer))
 
        return nn.Sequential(*layers)
 
    def forward(self, x): 
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x1 = self.layer1(x)
        x2 = self.layer2(self.maxpool1(x1))
        x3 = self.layer3(self.maxpool2(x2))
        x4 = self.layer4(self.maxpool3(x3))
        x5 = self.maxpool4(x4)
 
        return x1,x2,x3,x4,x5

 
class bn_conv(nn.Module):
    def __init__(self,num_init_features,output_channels,dilation=1):
        super(bn_conv,self).__init__()
        self.features = nn.Sequential(OrderedDict([ 
            ('conv0', nn.Conv2d(num_init_features,output_channels,kernel_size=3, stride=1,
                                padding=dilation, bias=False,dilation=dilation)), 
            ('norm0', nn.BatchNorm2d(output_channels)),
            ('relu0', nn.ReLU(inplace=True))          
    
        ]))
    def forward(self,x):
        x=self.features(x)
        return x

class bn_conv2(nn.Module):
    def __init__(self, initial_channels,output_channels,dilation=1):
        super(bn_conv2,self).__init__()
        self.conv1=bn_conv(initial_channels,output_channels,dilation)
        self.conv2=bn_conv(output_channels,output_channels,dilation)
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        return x 

class up_conv(nn.Module):
    def __init__(self,initial_channels,output_channels,dilation):
        super(up_conv,self).__init__()
        self.upsample=Upsample(2)
        self.conv=bn_conv(initial_channels,output_channels,dilation)
    def forward(self,x):
        x=self.conv(self.upsample(x))
        return x

class conv_11(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(conv_11,self).__init__()
        self.conv=nn.Conv2d(input_channels,output_channels,kernel_size=1,stride=1)
    def forward(self,x):
        x=self.conv(x)
        return x
        
class decoder(nn.Module):
    def __init__(self,input_channels,upsample_factor,dilation):
        super(decoder,self).__init__()
        self.conv1=bn_conv(input_channels,input_channels//2,dilation)
        self.conv2=bn_conv(input_channels//2,input_channels//4,dilation)

        #Multi-Scale-Supervision-Block  
        self.conv3=nn.Conv2d(input_channels//4,64,kernel_size=3,stride=1,padding=1)
        self.conv4=conv_11(64,1)
        self.upsample=Upsample(upsample_factor)
        #End Multi-Scale_Supervision-Block
    def forward(self,x):
        x=self.conv1(x)
        x1=self.conv2(x)
        #Use of Multi-Scale-Supervision-Block 
        x2=self.conv3(x1)    
        x2=self.conv4(self.upsample(x2))
        return x1,x2

class final_decoder(nn.Module):
    def __init__(self,input_channels,dilation):
        super(final_decoder,self).__init__()
        self.conv1=bn_conv(input_channels,input_channels//2,dilation)
        self.conv2=nn.Conv2d(input_channels//2,64,kernel_size=3,stride=1,padding=1)
        self.conv3=conv_11(64,1)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        return x


class D3MSU(nn.Module):
    def __init__(self,input_channels):
        super(D3MSU,self).__init__()
        self.encoder=ResNet(input_channels,BasicBlock,[3,3,3,3])
        self.bn_conv0=bn_conv2(512,512,8)

        self.decoder1=decoder(1024,8,8)
        self.decoder2=decoder(512,4,4)
        self.decoder3=decoder(256,2,2)
        self.decoder4=final_decoder(128,1)

        self.upconv1=nn.ConvTranspose2d(512,512,kernel_size=2,stride=2)
        self.upconv2=nn.ConvTranspose2d(256,256,kernel_size=2,stride=2)
        self.upconv3=nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.upconv4=nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)


    def forward(self,x):
        
        x1,x2,x3,x4,x5=self.encoder(x)
        
        x5=self.bn_conv0(x5)        

        x5=self.upconv1(x5)
        x_4,x_out4=self.decoder1(torch.cat((x5,x4),1))

        x_4=self.upconv2(x_4)
        x_3,x_out3=self.decoder2(torch.cat((x_4,x3),1))

        x_3=self.upconv3(x_3)
        x_2,x_out2=self.decoder3(torch.cat((x_3,x2),1))

        x_2=self.upconv4(x_2)
        x_out1=self.decoder4(torch.cat((x_2,x1),1))
        
        
        return torch.sigmoid(x_out1), torch.sigmoid(x_out2), torch.sigmoid(x_out3), torch.sigmoid(x_out4)
  
