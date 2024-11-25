import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detect = torch.load('detV4.pth').to(device)

'''
:param img: PIL Image
:param threshold: float, 0-1. The threshold for the mask.
A value lower than 0 means to repaint the whole image.
A value of 0 means to repaint wherever the model is uncertain if it's a watermark.
It can also be used to 'detoxify' the image.
A value of 0.1 means to repaint wherever the model is certain it's a watermark.
:param erosion: int, 0-inf. The size of the erosion structuring element.
:param dilation: int, 0-inf. The size of the dilation structuring element.
:return: PIL Image
'''


def fix_img(img0, threshold=0.5, erosion=2, dilation=6, mask_only=False):
    img = img0.convert('RGB')
    npa = np.array(img)
    blue = npa[:, :, 2]
    blue_torch = torch.tensor(blue).float().to(device)

    mask = detect(blue_torch.unsqueeze(0)).cpu().detach().squeeze().numpy()
    mask = np.where(mask < threshold, 0, 1)

    if erosion > 1:
        struct_elem = np.ones((erosion, erosion), dtype=bool)
        mask = binary_erosion(mask, structure=struct_elem, iterations=1)

    struct_elem = np.ones((dilation, dilation), dtype=bool)
    mask = binary_dilation(mask, structure=struct_elem, iterations=1)
    mask = 1 - mask

    if mask_only:
        return Image.fromarray((mask * 255).astype(np.uint8), 'L')

    masked_r = npa[:, :, 0] * mask
    masked_g = npa[:, :, 1] * mask
    masked_b = npa[:, :, 2] * mask
    fixed = np.stack([masked_r, masked_g, masked_b], axis=2)
    return Image.fromarray(fixed.astype(np.uint8), 'RGB')


img = Image.open('sample.png')
fixed = fix_img(img, threshold=0.3, erosion=1, dilation=2, mask_only=True)
fixed.show()
fixed.save('mask.jpg')
