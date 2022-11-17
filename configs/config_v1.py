import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class CFG():
    # training
    version = "baseline_2D_LSTM"
    backbone = "tf_efficientnet_b2_ns"
    pretrained = True
    head_type = "classify"
    hidden_dim_lstm = 256
    
    # hyperparameters
    liveness_threshold = 0.5

    # data
    height = 260
    width = 260
    num_frames = 3
    ext = "jpg"
    fold = 0

    # optimizer
    learning_rate = 1e-4
    batch_size = 4
    num_workers = 2
    num_epochs = 30

    # log and path
    train_data_path = None
    test_data_path = None
    save_weight_frequency = 1


CFG.train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            A.Resize(height=CFG.height, width=CFG.width, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(always_apply=True),
        ],
        p = 1.0,
    )

CFG.val_transforms = A.Compose(
        [
            A.Resize(height=CFG.height, width=CFG.width, always_apply=True),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(always_apply=True),
        ],
        p = 1.0,
    )