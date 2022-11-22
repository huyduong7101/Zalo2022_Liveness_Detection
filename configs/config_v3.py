import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class CFG():
    description = "Transform config"

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