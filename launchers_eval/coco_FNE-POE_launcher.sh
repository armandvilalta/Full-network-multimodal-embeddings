python eval_main.py \
--grad_clip 2 \
--load_from best \
--test_subset 1k \
--data_path ../data \
--decay_c 0.0 \
--maxlen_w 100 \
--max_epochs 200 \
--dispFreq 10 \
--abs True \
--dim_image 12416 \
--method order \
--img_norm True \
--optimizer adam \
--validFreq 100 \
--dim 2048 \
--batch_size 128 \
--encoder gru \
--lrate 0.0001 \
--data coco \
--loss MOE \
--dim_word 1024 \
--model_name coco_FNE-POE \
--experiment_name FNE-POE \
--embedding AVGtt_FN_KSBsp0.15n0.25_Gall \
--save_dir ../trained_models_paper/coco \
--margin 0.05 \
--reload_ True \
--dataset_name coco