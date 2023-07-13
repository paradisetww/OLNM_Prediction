python train/train_resnet.py \
--gpu=0 \
--data_dir='/.../datasets/image_mask_new/' \
--data_name='resnet18_pretrain' \
--json_list='training_validation.json' \
--num=1 \
--RandRotate90d_prob=0.5 \
--batch_size=12 \
--n_epoch=5 \
--val_every=1 \
--cont_loss \
--lr=1e-4 \
--optimizer='Adam' \
--cos_atten \
--model_name='resnet_18' \
--pretrained

python test/test_resnet.py \
--gpu=0 \
--data_dir='/.../datasets/image_mask_new/' \
--data_name='resnet18_pretrain' \
--json_list='test_1.json' \
--batch_size=12 \
--model_name='resnet_18'

python test/test_resnet.py \
--gpu=0 \
--data_dir='/.../datasets/image_mask_new/' \
--data_name='resnet18_pretrain' \
--json_list='test_2.json' \
--batch_size=12 \
--model_name='resnet_18'