import os
import cv2
import torch
import numpy as np
from model.UNet import UNet
import torchvision.transforms.functional as TF
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import InterpolationMode


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
MODEL_PATH = "checkpoints/best_model.pth"
INPUT_IMAGE = "dataset/isic/ISBI2016_ISIC_Part1_Test_Data/ISIC_0000111.jpg"
OUTPUT_MASK = "images/ISIC_0000111_pred_mask.png"
OUTPUT_OVERLAY = "images/ISIC_0000111_pred_overlay.png"


def load_model(model_path, device):
    model = UNet(
        in_channels=3,
        out_channels=1,
        bn=True
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_image(image_path, image_size):
    image = read_image(image_path, mode=ImageReadMode.RGB).float() / 255.0
    orig_h, orig_w = image.shape[1], image.shape[2]

    image_resized = TF.resize(
        image,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True
    )

    return image, image_resized.unsqueeze(0), orig_h, orig_w


@torch.no_grad()
def predict(model, image_tensor, device, threshold=0.5):
    image_tensor = image_tensor.to(device)

    logits = model(image_tensor)
    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()

    pred = pred.squeeze(0).squeeze(0).cpu().numpy()
    prob = prob.squeeze(0).squeeze(0).cpu().numpy()

    return pred, prob


def save_results(orig_image, pred_mask, orig_h, orig_w, out_mask, out_overlay):
    pred_mask = cv2.resize(
        pred_mask.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )

    mask_vis = pred_mask * 255
    cv2.imwrite(out_mask, mask_vis)

    image_np = (orig_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    overlay = image_bgr.copy()
    overlay[pred_mask == 1] = (0, 0, 255)

    blended = cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)
    cv2.imwrite(out_overlay, blended)


def main():
    model = load_model(MODEL_PATH, DEVICE)
    orig_image, input_tensor, orig_h, orig_w = preprocess_image(INPUT_IMAGE, IMAGE_SIZE)
    pred_mask, _ = predict(model, input_tensor, DEVICE, threshold=0.5)
    save_results(orig_image, pred_mask, orig_h, orig_w, OUTPUT_MASK, OUTPUT_OVERLAY)

    print(f"Saved mask to: {OUTPUT_MASK}")
    print(f"Saved overlay to: {OUTPUT_OVERLAY}")


if __name__ == "__main__":
    main()