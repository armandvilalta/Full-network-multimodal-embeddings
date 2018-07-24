python eval_main.py \
--grad_clip 2 \
--load_from best \
--test_subset 1k \
--data_path ../data \
--decay_c 0.0 \
--maxlen_w 100 \
--max_epochs 100 \
--dispFreq 10 \
--abs True \
--dim_image 4096 \
--method order \
--img_norm True \
--optimizer adam \
--validFreq 100 \
--dim 1536 \
--batch_size 128 \
--encoder gru \
--lrate 0.001 \
--data coco \
--loss OE \
--dim_word 1024 \
--model_name coco_FC7-SOE \
--experiment_name FC7-SOE \
--embedding AVGtt_Gfc7 \
--save_dir ../trained_models_paper/coco \
--margin 0.05 \
--reload_ False \
--dataset_name coco