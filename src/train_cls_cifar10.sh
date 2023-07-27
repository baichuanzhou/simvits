python cls_cifar10.py --num_classes 10 \
--image_size 32 --patch_size 4 --in_channels 3 --ffn_dim 384 \
--depth 6 --n_head 12 --dropout 0.1 \
--embed_dim 384 --pool "cls" --patch_proj "conv" \
--pos_embed "cos" \
--fixed_patch_proj False \
--output_dir "output/5" \
--do_train True \
--do_eval True \
--do_predict True \
--optim "adamw" \
--logging_steps 100 \
--learning_rate 1e-3 \
--weight_decay 0.1 \
--lr_scheduler_type "cosine" \
--warmup_steps 1500 \
--epoch 100 \
--resume_from_checkpoint False \
--overwrite_output_dir True \

python cls_cifar10.py --num_classes 10 \
--image_size 32 --patch_size 4 --in_channels 3 --ffn_dim 384 \
--depth 6 --n_head 12 --dropout 0.1 \
--embed_dim 384 --pool="cls" --patch_proj="conv" \
--pos_embed "random" \
--fixed_patch_proj False \
--output_dir "output/2" \
--do_train True \
--do_eval True \
--do_predict True \
--optim "adamw" \
--logging_steps 100 \
--learning_rate 1e-3 \
--weight_decay 0.1 \
--lr_scheduler_type "cosine" \
--warmup_steps 1500 \
--epoch 100 \
--resume_from_checkpoint False \
--overwrite_output_dir False \

python cls_cifar10.py --num_classes 10 \
--image_size 32 --patch_size 4 --in_channels 3 --ffn_dim 384 \
--depth 6 --n_head 12 --dropout 0.1 \
--embed_dim 384 --pool "cls" --patch_proj "standard" \
--pos_embed "cos" \
--fixed_patch_proj True \
--output_dir "output/3" \
--do_train True \
--do_eval True \
--do_predict True \
--optim "adamw" \
--logging_steps 100 \
--learning_rate 1e-3 \
--weight_decay 0.1 \
--lr_scheduler_type "cosine" \
--warmup_steps 1500 \
--epoch=100 \
--resume_from_checkpoint False \
--overwrite_output_dir False \

python cls_cifar10.py --num_classes 10 \
--image_size 32 --patch_size 4 --in_channels 3 --ffn_dim 384 \
--depth 6 --n_head 12 --dropout 0.1 \
--embed_dim 384 --pool "cls" --patch_proj "standard" \
--pos_embed "random" \
--fixed_patch_proj True \
--output_dir "output/4" \
--do_train True \
--do_eval True \
--do_predict True \
--optim "adamw" \
--logging_steps 100 \
--learning_rate 1e-3 \
--weight_decay 0.1 \
--lr_scheduler_type "cosine" \
--warmup_steps 1500 \
--epoch 100 \
--resume_from_checkpoint False \
--overwrite_output_dir False \



python cls_cifar10.py --num_classes 10 \
--image_size 32 --patch_size 4 --in_channels 3 --ffn_dim 384 \
--depth 6 --n_head 12 --dropout 0.1 \
--embed_dim 384 --pool "mean" --patch_proj "standard" \
--pos_embed "cos" \
--fixed_patch_proj False \
--output_dir "output/5" \
--do_train True \
--do_eval True \
--do_predict True \
--optim "adamw" \
--logging_steps 100 \
--learning_rate 1e-3 \
--weight_decay 0.1 \
--lr_scheduler_type "cosine" \
--warmup_steps 1500 \
--epoch 100 \
--resume_from_checkpoint False \
--overwrite_output_dir False \


python cls_cifar10.py --num_classes 10 \
--image_size 32 --patch_size 4 --in_channels 3 --ffn_dim 384 \
--depth 6 --n_head 12 --dropout 0.1 \
--embed_dim 384 --pool "mean" --patch_proj "conv" \
--pos_embed "random" \
--fixed_patch_proj False \
--output_dir "output/6" \
--do_train True \
--do_eval True \
--do_predict True \
--optim "adamw" \
--logging_steps 100 \
--learning_rate 1e-3 \
--weight_decay 0.1 \
--lr_scheduler_type "cosine" \
--warmup_steps 1500 \
--epoch 100 \
--resume_from_checkpoint False \
--overwrite_output_dir False \


python cls_cifar10.py --num_classes 10 \
--image_size 32 --patch_size 4 --in_channels 3 --ffn_dim 384 \
--depth 6 --n_head 12 --dropout 0.1 \
--embed_dim 384 --pool "mean" --patch_proj "standard" \
--pos_embed "cos" \
--fixed_patch_proj True \
--output_dir "output/7" \
--do_train True \
--do_eval True \
--do_predict True \
--optim "adamw" \
--logging_steps 100 \
--learning_rate 1e-3 \
--weight_decay 0.1 \
--lr_scheduler_type "cosine" \
--warmup_steps 1500 \
--epoch 100 \
--resume_from_checkpoint False \
--overwrite_output_dir False


python cls_cifar10.py --num_classes 10 \
--image_size 32 --patch_size 4 --in_channels 3 --ffn_dim 384 \
--depth 6 --n_head 12 --dropout 0.1 \
--embed_dim 384 --pool "mean" --patch_proj "standard" \
--pos_embed "random" \
--fixed_patch_proj True \
--output_dir "output/8" \
--do_train True \
--do_eval True \
--do_predict True \
--optim "adamw" \
--logging_steps 100 \
--learning_rate 1e-3 \
--weight_decay 0.1 \
--lr_scheduler_type "cosine" \
--warmup_steps 1500 \
--epoch 100 \
--resume_from_checkpoint False \
--overwrite_output_dir False