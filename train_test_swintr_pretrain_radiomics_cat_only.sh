python train/train_swintr_radiomics.py \
--gpu=0 \
--data_dir='/.../datasets/image_mask_new/' \
--data_name='swintr_pretrain_radiomics_cat_only' \
--json_list='training_validation.json' \
--num=1 \
--RandRotate90d_prob=0.5 \
--batch_size=6 \
--n_epoch=5 \
--val_every=1 \
--cont_loss \
--lr=1e-4 \
--optimizer='Adam' \
--cos_atten \
--use_ssl_pretrained \
--fused_pattern='cat_only'

python test/test_swintr_radiomics.py \
--gpu=0 \
--data_dir='/.../datasets/image_mask_new/' \
--data_name='swintr_pretrain_radiomics_cat_only' \
--json_list='test_1.json' \
--batch_size=6 \
--fused_pattern='cat_only'

python test/test_swintr_radiomics.py \
--gpu=0 \
--data_dir='/.../datasets/image_mask_new/' \
--data_name='swintr_pretrain_radiomics_cat_only' \
--json_list='test_2.json' \
--batch_size=6 \
--fused_pattern='cat_only'
