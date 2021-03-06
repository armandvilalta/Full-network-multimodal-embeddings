python eval_main.py \
--grad_clip 2 \
--load_from best \
--test_subset 1k \
--data_path ../data \
--decay_c 0.0 \
--maxlen_w 100 \
--max_epochs 400 \
--dispFreq 10 \
--abs False \
--dim_image 4096 \
--method cosine \
--img_norm True \
--optimizer adam \
--validFreq 100 \
--dim 1024 \
--batch_size 128 \
--encoder gru \
--lrate 0.0002 \
--data f8k \
--loss MH \
--dim_word 300 \
--model_name f8k_FC7-MH-bl \
--experiment_name FC7-MH-bl \
--embedding AVGtt_Gfc7 \
--save_dir ../trained_models_paper/f8k \
--margin 0.2 \
--reload_ True \
--dataset_name f8k