import pickle

from dlutils.pytorch import count_parameters
from dlutils import download
from tqdm import tqdm

from PIL import Image
from matplotlib import pyplot as plt
import lpips

class LPIPS:
    '''
    borrowed from https://github.com/richzhang/PerceptualSimilarity
    '''
    def __init__(self, cuda, des="Learned Perceptual Image Patch Similarity", version="0.1"):
        self.des = des
        self.version = version
        self.model = lpips.PerceptualLoss(model='net-lin',net='alex',use_gpu=cuda)

    def __repr__(self):
        return "LPIPS"

    def __call__(self, y_pred, y_true, normalized=True):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            normalized : change [0,1] => [-1,1] (default by LPIPS)
        return LPIPS, smaller the better
        """
        if normalized:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        return self.model.forward(y_pred, y_true)


def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=None,
        truncation_cutoff=None,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)

    model.cuda(0)
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder

    mapping_fl = model.mapping_fl
    dlatent_avg = model.dlatent_avg

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        'discriminator_s': encoder,
        # 'encoder_s': encoder,
        'generator_s': decoder,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg_s': dlatent_avg
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    extra_checkpoint_data = checkpointer.load()
    last_epoch = list(extra_checkpoint_data['auxiliary']['scheduler'].values())[0]['last_epoch']
    logger.info("Model trained for %d epochs" % last_epoch)

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    logger.info("Evaluating LPIPS metric")

    decoder = nn.DataParallel(decoder)
    encoder = nn.DataParallel(encoder)

    with torch.no_grad():
        ppl = LPIPS(cfg, num_images=10000, minibatch_size=16 * torch.cuda.device_count())
        ppl.evaluate(logger, mapping_fl, decoder, encoder, cfg.DATASET.MAX_RESOLUTION_LEVEL - 2)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-lpips', default_config='configs/experiment_celeba.yaml',
        world_size=gpu_count, write_log="metrics/lpips_score.txt")
