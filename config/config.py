alexnet = {
    'pre_trained_weights_fp': 'data/bvlc_alexnet.npy',
    'tensorboard_dir': 'data/alexnet_finetune/train',
    'num_classes': 1,
    'fine_tune_layers': ['fc8', 'fc7', 'fc6'],
    'model_save_path': 'data/alexnet_finetune/save/alexnet_finetune',
    'model_save_dir': 'data/alexnet_finetune/save',
    'img_csv_col_name': 'img',
    'csv_fp': 'data/alexnet_finetune/train.csv',
    'hyperparams': {
        'batch_size': 128,
        'img_height': 227,
        'img_width': 227,
        'num_channels': 3,
        'learning_rate': 0.0001,
        'num_epochs': 1000,
        'keep_prob': 0.5,
        'validation_period': 1,
        'test_period': 50,
        'data_split_ratios': [0.7, 0.2, 0.1]
    }
}