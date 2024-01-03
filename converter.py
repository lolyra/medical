import torch
from torch import nn
from timm.layers import SelectAdaptivePool2d
from timm.models.efficientvit_mit import LiteMLA

def load_state_dict_to_3d(model_3d, state_dict_2d, repeat_axis):
    present_dict = model_3d.state_dict()
    for key in list(state_dict_2d.keys()):
        if state_dict_2d[key].dim()<present_dict[key].dim():
            repeat_times = present_dict[key].shape[repeat_axis]
            state_dict_2d[key] = torch.stack([state_dict_2d[key]]*repeat_times, dim=repeat_axis) / repeat_times
    return model_3d.load_state_dict(state_dict_2d, strict=True)

def convert_layer_to_3d(layer):
    if isinstance(layer,nn.Conv2d):
        return nn.Conv3d(
            layer.in_channels, 
            layer.out_channels,
            groups=layer.groups,
            kernel_size=layer.kernel_size[0],
            stride=layer.stride[0], 
            padding=layer.padding[0],
            dilation=layer.dilation[0],
            padding_mode=layer.padding_mode,
            bias=layer.bias is not None
        )
    elif isinstance(layer,nn.BatchNorm2d):
        return nn.BatchNorm3d(
            layer.num_features,
            eps=layer.eps,
            momentum=layer.momentum,
            affine=layer.affine,
            track_running_stats=layer.track_running_stats,
        )
    elif isinstance(layer,SelectAdaptivePool2d):
        return nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size=1),
            nn.Flatten(1)
        )
    return layer

def convert_module_to_3d(model):
    for key in model._modules:
        module = model._modules[key]
        if '_modules' in module.__dict__:
            model._modules[key] = convert_module_to_3d(model._modules[key])
        model._modules[key] = convert_layer_to_3d(model._modules[key])
    return model

def forward_3d(self, x):
    B, _, D, H, W = x.shape

    # generate multi-scale q, k, v
    qkv = self.qkv(x)
    multi_scale_qkv = [qkv]
    for op in self.aggreg:
        multi_scale_qkv.append(op(qkv))
    multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)
    multi_scale_qkv = multi_scale_qkv.reshape(B, -1, 3 * self.dim, D * H * W).transpose(-1, -2)
    q, k, v = multi_scale_qkv.chunk(3, dim=-1)

    # lightweight global attention
    q = self.kernel_func(q)
    k = self.kernel_func(k)
    v = nn.functional.pad(v, (0, 1), mode="constant", value=1.)

    if not torch.jit.is_scripting():
        with torch.autocast(device_type=v.device.type, enabled=False):
            out = self._attn(q, k, v)
    else:
        out = self._attn(q, k, v)

    # final projection
    out = out.transpose(-1, -2).reshape(B, -1, D, H, W)
    out = self.proj(out)
    return out

def convert_model_to_3d(model, repeat_axis=-1):
    state_dict_2d = model.state_dict()
    convert_module_to_3d(model)
    load_state_dict_to_3d(model, state_dict_2d, repeat_axis)
    for i in [2,3]:
        for j in range(1,len(model.stages[i].blocks)):
            model.stages[i].blocks[j].context_module.main.forward = forward_3d.__get__(model.stages[i].blocks[j].context_module.main,LiteMLA)
    return model


