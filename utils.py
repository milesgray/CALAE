import pathlib

from tqdm import tqdm
import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
from torchvision.utils import save_image

def ensure_dir(path, verbose=False):
    path = pathlib.Path(path)
    if not path.exists():
        path.mkdir(parents=True)
        if verbose: print(f"[INFO]\t Created {path.absolute()}")
    else:
        if verbose: print(f"[INFO]\t Using {path.absolute()}")
    return path

def ensure_parent_dir(path, verbose=False):
    path = pathlib.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
        if verbose: print(f"[INFO]\t Created {path.absolute()}")
    else:
        if verbose: print(f"[INFO]\t Using {path.absolute()}")
    return path

def sample_noise(bs, code=512, device='cpu'):
    return torch.randn(bs, code).to(device)

def find_alpha(tracked, limit):
    return min(tracked/max(limit, 1), 1)

def allow_gradient(module, permission=True):
    for block in module.parameters():
        block.requires_grad = permission

def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult
        
def linear_scale_lr(tracked, total_items, start=5e-6, end=1.5e-4):
    coef = tracked/total_items
    return (1 - coef) * start + coef * end

def get_total_elements(obj, verbose=False):
    try:
        if torch.is_tensor(obj):
            total = fake.size()[0]
        elif isinstance(obj, tuple):
            total = len(fake)
        elif isinstance(obj, np.ndarray):
            total = fake.shape[0]

        return total
    except Exception as e:
        if verbose: print(f"[ERROR]\t get_total_elements:: {e}")
        return 0

def save_batch(name, fake, real, nrows=6, split=(4,2), verbose=False):
    fake_total = get_total_elements(fake)
    real_total = get_total_elements(real)
    if verbose: print(f"Saving: fake: {fake_total} real: {real_total}")
    try:

        fake, real = fake.split(split[0]), real.split(split[1])
        save_image(torch.cat([torch.cat([fake[i], real[i]], dim=0) for i in range(nrows)], dim=0), name, nrow=nrows, padding=1,
                normalize=True, range=(-1, 1))
    except Exception as e:
        if verbose: print(f"[ERROR]\t Couldn't save! shape fake: {len(fake)}, shape real: {len(real)}, nrows: {nrows} \n\t\t{e}")
    
def save_reconstructions(name, original, reconstruction, nrows=6):
    """
    original, reconstruction - type: list, e.g. original = [x, x_hat], reconstruction = [G(E(x)), G(E(x_hat))]
    
    [[orig_x, rec_x], [orig_x, rec_x], [orig_x, rec_x]]
    [[orig_x_hat, rec_x_hat], [orig_x_hat, rec_x_hat], [orig_x_hat, rec_x_hat]]
    
    """
    tensor = []
    for orig, rec in zip(original, reconstruction):        
        tensor.append(torch.cat([torch.cat([orig.split(1)[i], rec.split(1)[i]], dim=0) for i in range(nrows//2)], dim=0))
    
    save_image(torch.cat(tensor, dim=0), name, nrow=nrows, padding=1, normalize=True, range=(-1, 1))


def img_from_tensor(x):
    if len(x.shape) > 3:
        x = x.squeeze().cpu().detach()
    return ((x * 0.5 + 0.5) * 255) \
                .type(torch.long) \
                    .clamp(0, 255) \
                        .cpu() \
                            .type(torch.uint8) \
                                .transpose(0, 2) \
                                    .transpose(0, 1) \
                                        .numpy()

# ======================= #
# Video-related functions #
# ======================= #


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets, args, x_src, s_prev, s_next):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    alphas = get_alphas()

    for alpha in alphas:
        s_ref = torch.lerp(s_prev, s_next, alpha)
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        entries = torch.cat([x_src.cpu(), x_fake.cpu()], dim=2)
        frame = torchvision.utils.make_grid(entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next, 
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas) # number of frames

    canvas = - torch.ones((T, C, H*2, W + margin))
    merged = torch.cat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


@torch.no_grad()
def video_ref(nets, args, x_src, x_ref, y_ref, fname):
    video = []
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_prev = None
    for data_next in tqdm(zip(x_ref, y_ref, s_ref), 'video_ref', len(x_ref)):
        x_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
        if s_prev is None:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue
        if y_prev != y_next:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue

        interpolated = interpolate(nets, args, x_src, s_prev, s_next)
        entries = [x_prev, x_next]
        slided = slide(entries)  # (T, C, 256*2, 256)
        frames = torch.cat([slided, interpolated], dim=3).cpu()  # (T, C, 256*2, 256*(batch+1))
        video.append(frames)
        x_prev, y_prev, s_prev = x_next, y_next, s_next

    # append last frame 10 time
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


@torch.no_grad()
def video_latent(nets, args, x_src, y_list, z_list, psi, fname):
    latent_dim = z_list[0].size(1)
    s_list = []
    for i, y_trg in enumerate(y_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(x_src.size(0), 1)

        for z_trg in z_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            s_list.append(s_trg)

    s_prev = None
    video = []
    # fetch reference images
    for idx_ref, s_next in enumerate(tqdm(s_list, 'video_latent', len(s_list))):
        if s_prev is None:
            s_prev = s_next
            continue
        if idx_ref % len(z_list) == 0:
            s_prev = s_next
            continue
        frames = interpolate(nets, args, x_src, s_prev, s_next).cpu()
        video.append(frames)
        s_prev = s_next
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


def save_video(fname, images, output_fps=30, vcodec='libx264', filters=''):
    assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
    num_frames, height, width, channels = images.shape
    stream = ffmpeg.input('pipe:', format='rawvideo', 
                          pix_fmt='rgb24', s='{}x{}'.format(width, height))
    stream = ffmpeg.filter(stream, 'setpts', '2*PTS')  # 2*PTS is for slower playback
    stream = ffmpeg.output(stream, fname, pix_fmt='yuv420p', vcodec=vcodec, r=output_fps)
    stream = ffmpeg.overwrite_output(stream)
    process = ffmpeg.run_async(stream, pipe_stdin=True)
    for frame in tqdm(images, desc='writing video to %s' % fname):
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255