# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import json
import argparse
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from networks.swintr_radiomics import Swin_Tr

import monai
from monai.data import load_decathlon_datalist, decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, ToTensord

parser = argparse.ArgumentParser(description="The pipeline of predicting lymph node metastasis")
parser.add_argument('--gpu', default=0, type=int, help='use gpu device')
parser.add_argument("--data_dir", default="./datasets/dataset0/", type=str, help="dataset directory")
parser.add_argument("--data_name", default="dataset0", type=str, help="dataset name")
parser.add_argument("--json_dir", default="./datasets/", type=str, help="dataset json directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--logdir", default="./output", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--batch_size", default=16, type=int, help="number of batch size")
parser.add_argument("--workers", default=3, type=int, help="number of workers")
parser.add_argument('--num_radiomics_features', type=int, default=10, help='number of radiomics features')
parser.add_argument('--fused_pattern', default="add", type=str, help='fused pattern of features')
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_ssl_pretrained", action="store_true", help="use self-supervised pretrained weights")


def main():
    args = parser.parse_args()

    data_dir = args.data_dir
    datalist_json = os.path.join(args.json_dir, args.json_list)
    args.logdir = os.path.join(args.logdir, args.data_name)

    # Define transforms for image
    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
            ToTensord(keys=["image", "mask"])
        ]
    )
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    # Define dataset, data loader
    test_files = load_decathlon_datalist(datalist_json, False, "test", base_dir=data_dir)

    # create a test data loader
    test_ds = monai.data.CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    print(f'There are {len(test_ds)} samples!')

    # Create Swin Transformer
    torch.cuda.set_device(args.gpu)
    model = Swin_Tr(args).cuda(args.gpu)
    
    model_path = os.path.join(args.logdir, "models/best_auc_model.pth")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    epoch = checkpoint["best_auc_epoch"]

    auc_metric = ROCAUCMetric()
    args.logdir = os.path.join(args.logdir, args.json_list[:-5])
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    start_time = time.time()
    
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32).cuda(args.gpu)
        y = torch.tensor([], dtype=torch.long).cuda(args.gpu)
        for test_data in tqdm(test_loader):
            test_data["radiomics"] = torch.stack(test_data["radiomics"]).transpose(0, 1).float()
            test_images, test_labels, test_radiomics_features = test_data["image"].cuda(args.gpu), test_data["label"].cuda(args.gpu), test_data["radiomics"].cuda(args.gpu)
            y_pred = torch.cat([y_pred, model(test_images, test_radiomics_features)[0]], dim=0)
            y = torch.cat([y, test_labels], dim=0)
        
        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
        y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]

        auc_metric(y_pred_act, y_onehot)
        auc_result = auc_metric.aggregate()
        auc_metric.reset()
        
        gt = y.detach().cpu().numpy().reshape(-1,1)
        pred = np.zeros((len(y_pred_act),1))
        for idx in range(len(y_pred_act)):
            pred[idx][0] = y_pred_act[idx][1].item()

        del y_pred_act, y_onehot

    with open(os.path.join(args.logdir, args.json_list), "w") as f:
        acc_metric = float(acc_metric)
        auc_result = float(auc_result)
        json.dump({"epoch": epoch, "acc": acc_metric, "auc": auc_result}, f)

    result = np.concatenate((gt, pred), axis=1)
    result_df = pd.DataFrame(result)
    result_df.columns = ['group', 'prediction']
    writer = pd.ExcelWriter(os.path.join(args.logdir, args.json_list[:-5] + '.xlsx'))
    result_df.to_excel(writer, index=False, header=True, float_format='%.4f')
    writer.save()
    writer.close()

    print(f"Test completed, epoch: {epoch}, accuracy: {acc_metric:.4f}, auc: {auc_result:.4f}, time: {time.time() - start_time:.2f}s!")


if __name__ == "__main__":
    main()
