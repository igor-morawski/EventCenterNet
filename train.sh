python train.py --data_dir=/tmp2/igor/EV/Dataset/Automotive/ --log_name=test --train_csv_file=/tmp2/igor/EV/EventCenterNet/datasets/train_a.csv --val_csv_file=/tmp2/igor/EV/EventCenterNet/datasets/val_a.csv --class_list_file=/tmp2/igor/EV/EventCenterNet/datasets/classes.csv --trim_to_shortest --delta_t=50000 --frames_per_batch=5 --bins=5 --batch_size=2 --val_interval=1 --log_interval=1