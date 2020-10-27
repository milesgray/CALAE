import torch
from torch import nn
from torch.nn import functional as F

eps = 1e-6

def _binarize(y_data, threshold):
    """
    args:
        y_data : [float] 4-d tensor in [batch_size, channels, img_rows, img_cols]
        threshold : [float] [0.0, 1.0]
    return 4-d binarized y_data
    """
    y_data[y_data < threshold] = 0.0
    y_data[y_data >= threshold] = 1.0
    return y_data


class PSNRBinary(object):
    def __init__(self, des="Peak Signal to Noise Ratio"):
        self.des = des

    def __repr__(self):
        return "PSNR"

    def __call__(self, y_pred, y_true, dim=1, threshold=None):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return PSNR, larger the better
        """
        if threshold:
            y_pred = _binarize(y_pred, threshold)
        mse = torch.mean((y_pred - y_true) ** 2)
        return 10 * torch.log10(1 / mse)

class PSNR(nn.Module):
    def __init__(self, max_value: int = 1):
        super().__init__()
        self.name = "PSNR"
        self.max_value = max_value

    def forward(self, image_1: torch.Tensor, image_2: torch.Tensor) -> torch.Tensor:
        """ Peak Signal to Noise Ratio
        args:
            image_1 : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            image_2 : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return PSNR, larger the better
        """
        assert image_1.shape == image_2.shape, "For a meaningful PSNR calculation, the shape of image_1 and image_2 should be the same"

        if len(image_1.shape) == 4:
            # we are dealing with a batch of images
            reduction = 'none'
            mean_dims = (1, 2, 3)
        else:
            reduction = 'mean'
            mean_dims = None

        mse = F.mse_loss(image_1, image_2, reduction=reduction)
        if mean_dims is not None:
            mse = mse.mean(dim=mean_dims)

        psnr = 20 * torch.log10(self.max_value ** 2 / torch.sqrt(mse))
        return psnr.mean()


