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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pytorch_metric_learning import losses
from utils.utils import AverageMeter, create_logger, adjust_learning_rate, ModelEma
from networks.resnet import resnet18, resnet34, resnet50

import monai
from monai.data import load_decathlon_datalist, decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandCropByPosNegLabeld, RandRotate90d, ToTensord

parser = argparse.ArgumentParser(description="The pipeline of predicting lymph node metastasis")
parser.add_argument('--gpu', default=0, type=int, help='use gpu device')
parser.add_argument("--data_dir", default="./datasets/dataset0/", type=str, help="dataset directory")
parser.add_argument("--data_name", default="dataset0", type=str, help="dataset name")
parser.add_argument("--json_dir", default="./datasets/", type=str, help="dataset json directory")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--logdir", default="./output", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--num", default=1, type=int, help="number of random crop")
parser.add_argument("--RandRotate90d_prob", default=0.8, type=float, help="RandRotate90d aug probability")
parser.add_argument("--batch_size", default=16, type=int, help="number of batch size")
parser.add_argument("--workers", default=3, type=int, help="number of workers")
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--n_epoch', default=5, type=int, help='number of epoch to change')
parser.add_argument("--max_epochs", default=100, type=int, help="max number of training epochs")
parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
parser.add_argument('--seed', type=int, default=5, help='random seed.')
parser.add_argument('--cont_loss', action='store_true', help='contrastive loss')
parser.add_argument("--lr", default=1e-4, type=float, help="optimization learning rate")
parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer (Adam)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, help='weight decay of optimizer (default: 1e-4)')
parser.add_argument('--cos_atten', action='store_true', help='cosine attenuation')
parser.add_argument('--resume', default='./output/file_name/checkpoint/', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--cls_weights', action='store_true', help='class weights of lnm')
parser.add_argument("--model_name", default="resnet_50", type=str, help="model name")
parser.add_argument("--pretrained", action="store_true", help="use pretrained weights")


def main():
    args = parser.parse_args()

    # Fixed random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)

    data_dir = args.data_dir
    datalist_json = os.path.join(args.json_dir, args.json_list)

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
            RandCropByPosNegLabeld(
                keys=["image", "mask"],
                label_key="mask",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=0,
                num_samples=args.num,
                image_key="image",
                image_threshold=0,
            ),
            RandRotate90d(keys=["image", "mask"], prob=args.RandRotate90d_prob, spatial_axes=[0, 2], max_k=3),
            ToTensord(keys=["image", "mask"])
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
            ToTensord(keys=["image", "mask"])
        ]
    )
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    # Define dataset, data loader
    train_files = load_decathlon_datalist(datalist_json, False, "train", base_dir=data_dir)
    val_files = load_decathlon_datalist(datalist_json, False, "validation", base_dir=data_dir)

    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print(check_data["image"].shape, check_data["mask"].shape, check_data["label"])
    print("Batch size is:", args.batch_size, "epochs", args.max_epochs)

    # create a training data loader
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=args.workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=torch.cuda.is_available())

    best_auc = -1
    best_auc_epoch = -1
    best_auc_acc = -1
    best_pred = np.zeros((len(val_ds),1)) 

    # Create model, CrossEntropyLoss and Adam optimizer
    torch.cuda.set_device(args.gpu)
    if args.model_name == "resnet_18":
        model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).cuda(args.gpu)
    elif args.model_name == "resnet_34":
        model = resnet34(spatial_dims=3, n_input_channels=1, num_classes=2).cuda(args.gpu)
    elif args.model_name == "resnet_50":
        model = resnet50(spatial_dims=3, n_input_channels=1, num_classes=2).cuda(args.gpu)
    else:
        raise RuntimeError(f"Model {args.model_name} is not found!")
    
    if args.pretrained:
        try:
            model_dict = torch.load(f"./models/{args.model_name}_23dataset.pth")
            state_dict = model_dict["state_dict"]
            # fix potential differences in state dict keys from pre-training to
            # fine-tuning
            if "module." in list(state_dict.keys())[0]:
                print("Tag 'module.' found in state dict - fixing!")
                for key in list(state_dict.keys()):
                    state_dict[key.replace("module.", "")] = state_dict.pop(key)
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)
            print("Using pretrained backbone weights!")
        except ValueError:
                raise ValueError("Pre-trained weights are not available!")

    if os.path.exists(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["state_dict"])
        best_auc = checkpoint["best_auc"]
        best_auc_epoch = checkpoint["best_auc_epoch"]
        best_auc_acc = checkpoint["best_auc_acc"]

    if args.cls_weights:
        weights = [4, 1]
        class_weights = torch.FloatTensor(weights).cuda(args.gpu)
        loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_function = torch.nn.CrossEntropyLoss()
        
    auc_metric = ROCAUCMetric()
    auc_metric_ema = ROCAUCMetric()

    # define optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    else:
        print('Please choose true optimizer.')
        return 0

    # start a typical PyTorch training
    args.logdir = os.path.join(args.logdir, args.data_name)
    writer = SummaryWriter(args.logdir)
    logger, _ = create_logger(args)
    print("Writing logs to ", args.logdir)
    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    logger.info("-------Training started-------")
    start_time = time.time()
    
    model_path = os.path.join(args.logdir, "models")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82

    for epoch in range(args.start_epoch, args.max_epochs):
        epoch_time = time.time()
        train_losses = AverageMeter()
        if args.cos_atten:
            optimizer, writer = adjust_learning_rate(args, optimizer, epoch, writer)
        epoch_len = len(train_loader)
        model.train()
        for step, batch_data in enumerate(train_loader):
            step_time = time.time()
            inputs, labels = batch_data["image"].cuda(args.gpu), batch_data["label"].cuda(args.gpu)
            optimizer.zero_grad()
            outputs, features = model(inputs)

            if args.cont_loss:
                temperature = 0.05
                cont_loss_func = losses.NTXentLoss(temperature)
                cont_loss = cont_loss_func(features, labels)
                train_loss = loss_function(outputs, labels) + cont_loss
            else:
                train_loss = loss_function(outputs, labels)

            train_losses.update(train_loss.item(), inputs.size(0))
            train_loss.backward()
            optimizer.step()
            ema.update(model)
            print(f"Epoch {epoch}/{args.max_epochs}, Batch: {step}/{epoch_len}, train_loss: {train_losses.val:.4f}({train_losses.avg:.4f}), time: {time.time() - step_time:.2f}s")
        writer.add_scalar("train_loss", train_losses.avg, epoch)
        pbar_str = "---Training_Epoch:{}/{}  Loss:{:.4f}  Time:{:.2f}s".format(epoch, args.max_epochs, train_losses.avg, time.time() - epoch_time)
        logger.info(pbar_str)

        if (epoch + 1) % args.val_every == 0:
            epoch_time = time.time()
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32).cuda(args.gpu)
                y_pred_ema = torch.tensor([], dtype=torch.float32).cuda(args.gpu)
                y = torch.tensor([], dtype=torch.long).cuda(args.gpu)
                for val_data in tqdm(val_loader):
                    val_images, val_labels = val_data["image"].cuda(args.gpu), val_data["label"].cuda(args.gpu)
                    y_pred = torch.cat([y_pred, model(val_images)[0]], dim=0)
                    y_pred_ema = torch.cat([y_pred_ema, ema.module(val_images)[0]], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_value_ema = torch.eq(y_pred_ema.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                acc_metric_ema = acc_value_ema.sum().item() / len(acc_value_ema)
                y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                y_pred_act_ema = [post_pred(i) for i in decollate_batch(y_pred_ema)]

                auc_metric(y_pred_act, y_onehot)
                auc_metric_ema(y_pred_act_ema, y_onehot)
                auc_result = auc_metric.aggregate()
                auc_result_ema = auc_metric_ema.aggregate()
                auc_metric.reset()
                auc_metric_ema.reset()

                gt = y.detach().cpu().numpy().reshape(-1,1)
                pred = np.zeros((len(y_pred_act),1))
                pred_ema = np.zeros((len(y_pred_act_ema),1))
                for idx in range(len(y_pred_act)):
                    pred[idx][0] = y_pred_act[idx][1].item()
                for idx in range(len(y_pred_act_ema)):
                    pred_ema[idx][0] = y_pred_act_ema[idx][1].item()

                del y_pred_act, y_pred_act_ema, y_onehot

                auc_max = max(auc_result, auc_result_ema)
                if auc_result_ema >= auc_result:
                    ema_best = True
                    acc_max = acc_metric_ema
                    pred_max = pred_ema
                else:
                    ema_best = False
                    acc_max = acc_metric
                    pred_max = pred

                if auc_max > best_auc:
                    best_auc = auc_max
                    best_auc_epoch = epoch
                    best_auc_acc = acc_max
                    best_pred = pred_max
                    try:
                        if ema_best:
                            save_dict = {"state_dict": ema.module.state_dict(), "best_auc": best_auc, "best_auc_epoch": best_auc_epoch, "best_auc_acc": best_auc_acc}
                        else:
                            save_dict = {"state_dict": model.state_dict(), "best_auc": best_auc, "best_auc_epoch": best_auc_epoch, "best_auc_acc": best_auc_acc}
                        filename = os.path.join(model_path, "best_auc_model.pth")
                        torch.save(save_dict, filename)
                        print("Saved new best auc model!")
                    except:
                        print('store failed')
                        pass

                pbar_str = "---Validation_Epoch:{}/{}  Accuracy:{:.4f}  AUC:{:.4f}  Time:{:.2f}s".format(epoch, args.max_epochs, acc_max, auc_max, time.time() - epoch_time)
                logger.info(pbar_str)
                pbar_str = "---Best_AUC_Epoch:{}  Best_Acc:{:.4f}  Best_AUC:{:.4f}".format(best_auc_epoch, best_auc_acc, best_auc)
                logger.info(pbar_str)
                writer.add_scalar("val_acc", acc_max, epoch)
                writer.add_scalar("val_auc", auc_max, epoch)
    
    with open(os.path.join(args.logdir, "val_best_auc.json"), "w") as f:
        best_auc_acc = float(best_auc_acc)
        best_auc = float(best_auc)
        json.dump({"best_auc_epoch": best_auc_epoch, "best_acc": best_auc_acc, "best_auc": best_auc}, f)

    result = np.concatenate((gt, best_pred), axis=1)
    result_df = pd.DataFrame(result)
    result_df.columns = ['group', 'prediction']
    writer_1 = pd.ExcelWriter(os.path.join(args.logdir, args.json_list[:-5] + '.xlsx'))
    result_df.to_excel(writer_1, index=False, header=True, float_format='%.4f')
    writer_1.save()
    writer_1.close()

    print(f"Train completed, best auc epoch: {best_auc_epoch}, best accuracy: {best_auc_acc:.4f}, best auc: {best_auc:.4f}, time: {time.time() - start_time:.2f}s!")
    writer.close()


if __name__ == "__main__":
    main()
