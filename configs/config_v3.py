import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class CFG():
    # training
    model_name = "2D_multiframes"
    backbone = "tf_efficientnetv2_s_in21ft1k"
    pretrained = True
    head_type = "classify"
    use_lstm = False
    
    # hyperparameters
    liveness_threshold = 0.5

    # data
    height = 384
    width = 384
    num_frames = 5
    ext = "jpg"
    fold = 0

    # optimizer
    learning_rate = 1e-4
    batch_size = 16
    num_workers = 2
    num_epochs = 20

    # log and path
    train_data_path = None
    test_data_path = None
    save_weight_frequency = 1
    comet_api_key = "MqbVRKXYTLalajpK9uSDwDtOk"
    comet_project_name = "Zalo2022_LivenessDetection"

CFG.train_transforms = A.Compose(
        [
            A.Resize(height=CFG.height, width=CFG.width, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            # A.RandomBrightness(limit=0.1, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.25, scale_limit=[0, 0.25], rotate_limit=30, border_mode=4, p=0.5),
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