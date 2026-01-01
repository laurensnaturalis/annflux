2025-12-23 10:16:22,453 train command:
-----
2025-12-23 10:16:22,453 screen -L -S train_all -Logfile $PROJECT_ROOT/logs/log_train_all.log python /app/Scripts/train_multiclass.py /mnt/big/storage/diopsis26/datarepo/dataset-20251223101622-c3971751/dataset.hdf5 model_for_dataset_c3971751e70502e2b7888a74a403225d3e992eccd09c61bf4f0bc9f7 --architecture efficientnetv2m --use_warmup --batch_size 16
2025-12-23 10:16:22,453 evaluate command:
-----
2025-12-23 10:16:22,453 screen -L -S eval_all -Logfile $PROJECT_ROOT/logs/log_eval_all.log python /app/Scripts/evaluate_model.py all '' --model_folder /mnt/big/storage/diopsis26/jobs/model_for_dataset_c3971751e70502e2b7888a74a403225d3e992eccd09c61bf4f0bc9f7
2025-12-23 10:16:22,453 tensorboard command:
-----
2025-12-23 10:16:22,453 screen -L -S tb_all -Logfile log_tb_all tensorboard --logdir /storage/diopsis26/jobs/model_for_dataset_c3971751e70502e2b7888a74a403225d3e992eccd09c61bf4f0bc9f7/logs --host 0.0.0.0 --path_prefix /tensorboard/GPU$CUDA_VISIBLE_DEVICES
