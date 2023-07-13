# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from typing import Type, Union

import torch
import torch.nn as nn

from monai.networks.layers.factories import Pool
from monai.networks.layers.utils import get_act_layer
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep

class Swin_Tr(nn.Module):
    def __init__(self, args, dim: int = 768, hidden_size: int = 1024, embedding_size: int = 128, act: Union[str, tuple] = ("relu", {"inplace": True})):
        super(Swin_Tr, self).__init__()
        avg_pool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            Pool.ADAPTIVEAVG, args.spatial_dims
        ]
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
        )
        if args.use_ssl_pretrained:
            try:
                model_dict = torch.load("./models/model_swinvit.pt")
                state_dict = model_dict["state_dict"]
                # fix potential differences in state dict keys from pre-training to
                # fine-tuning
                if "module." in list(state_dict.keys())[0]:
                    print("Tag 'module.' found in state dict - fixing!")
                    for key in list(state_dict.keys()):
                        state_dict[key.replace("module.", "")] = state_dict.pop(key)
                if "swin_vit" in list(state_dict.keys())[0]:
                    print("Tag 'swin_vit' found in state dict - fixing!")
                    for key in list(state_dict.keys()):
                        state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
                msg = self.swinViT.load_state_dict(state_dict, strict=False)
                print(msg)
                print("Using pretrained self-supervised Swin UNETR backbone weights!")
            except ValueError:
                raise ValueError("Self-supervised pre-trained weights are not available!")
        
        self.fused_pattern = args.fused_pattern

        # pooling
        self.pooling_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)),
                ]
            )
        )

        if self.fused_pattern == "cat_only":
            num_fused_features = args.num_radiomics_features + dim
        elif self.fused_pattern == "cat":
            self.radiomics_fc = nn.Linear(args.num_radiomics_features, dim)
            num_fused_features = dim * 2
        elif self.fused_pattern == "add":
            self.radiomics_fc = nn.Linear(args.num_radiomics_features, dim)
            num_fused_features = dim
        elif self.fused_pattern == "lr_add":
            self.weight1 = nn.Parameter(torch.Tensor([1.0]))  # 放射组学特征的权重
            self.weight2 = nn.Parameter(torch.Tensor([1.0]))  # 卷积神经网络特征的权重
            self.radiomics_fc = nn.Linear(args.num_radiomics_features, dim)
            num_fused_features = dim

        # classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("out", nn.Linear(num_fused_features, args.out_channels)),
                ]
            )
        )
        self.fc1 = nn.Linear(num_fused_features, args.out_channels)
        self.fc2 = nn.Linear(num_fused_features, args.out_channels)
        self.fc3 = nn.Linear(num_fused_features, args.out_channels)
        self.fc4 = nn.Linear(num_fused_features, 1)

        # contrastive features
        self.contrastive_features = nn.Sequential(
            OrderedDict(
                [
                    ("feature1", nn.Linear(num_fused_features, hidden_size)),
                    ("feature2", nn.Linear(hidden_size, embedding_size)),
                ]
            )
        )
    
    def forward(self, x: torch.Tensor, radiomics_features: torch.Tensor) -> torch.Tensor:
        x_out = self.swinViT(x.contiguous())[4]
        x = self.pooling_layers(x_out)

        if self.fused_pattern == "cat_only":
            fused_features = torch.cat((radiomics_features, x), dim=1)
        elif self.fused_pattern == "cat":
            radiomics_output = self.radiomics_fc(radiomics_features)
            fused_features = torch.cat((radiomics_output, x), dim=1)
        elif self.fused_pattern == "add":
            radiomics_output = self.radiomics_fc(radiomics_features)
            fused_features = radiomics_output + x
        elif self.fused_pattern == "lr_add":
            radiomics_output = self.radiomics_fc(radiomics_features)
            fused_features = self.weight1 * radiomics_output + self.weight2 * x

        logits = self.class_layers(fused_features)
        logits1 = self.fc1(fused_features)
        logits2 = self.fc2(fused_features)
        logits3 = self.fc3(fused_features)
        logits4 = self.fc4(fused_features)
        logits4 = torch.squeeze(logits4)
        cont_features = self.contrastive_features(fused_features)

        return logits, cont_features, logits1, logits2, logits3, logits4
