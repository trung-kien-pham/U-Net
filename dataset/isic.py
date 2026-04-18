import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.io import read_image, ImageReadMode


class ISICDataset(Dataset):
    def __init__(self, data_dir, split="train", image_transform=None, mask_transform=None, image_size=256):
        super().__init__()

        self.data_dir = data_dir
        self.split = split.lower()
        self.image_transform = image_transform
        self.image_size = image_size
        self.mask_transform = mask_transform

        if self.split == "train":
            self.images_dir = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Training_Data")
            self.masks_dir = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Training_GroundTruth")
        elif self.split == "test":
            self.images_dir = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_Data")
            self.masks_dir = os.path.join(data_dir, "ISBI2016_ISIC_Part1_Test_GroundTruth")
        else:
            raise ValueError("split must be 'train' or 'test'")

        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "*.jpg")))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"Không tìm thấy ảnh nào trong {self.images_dir}")

        self.ids = [
            os.path.basename(path).replace("ISIC_", "").replace(".jpg", "")
            for path in self.image_paths
        ]

    def __len__(self):
        return len(self.ids)

    def _get_image_path(self, image_id):
        return os.path.join(self.images_dir, f"ISIC_{image_id}.jpg")

    def _get_mask_path(self, image_id):
        return os.path.join(self.masks_dir, f"ISIC_{image_id}_segmentation.png")

    def __getitem__(self, index):
        image_id = self.ids[index]

        image_path = self._get_image_path(image_id)
        mask_path = self._get_mask_path(image_id)

        image = read_image(image_path, ImageReadMode.RGB).float() / 255.0
        mask = read_image(mask_path, ImageReadMode.GRAY).float() / 255.0

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST
        )

        mask = (mask > 0.5).float()

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return {
            "image": image.clone(),
            "mask": mask.clone(),
            "id": image_id
        }