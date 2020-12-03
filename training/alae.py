

from CALAE import losses
from CALAE import net
from CALAE import utils
import piq
import lpips

def train(models, args, hyperparams, loss_dict, loss_stats, **kwargs):
    for scale_index, scale in enumerate(hyperparams["valid_scales"]):
        scale_num += 1
        epochs = hyperparams["epochs_by_scale"][scale]
        batch_size = hyperparams["batch_size_by_scale"][scale]
        args["SAVE_IMAGES_EACH"] = 25 if scale > 64 else 50
        args["SAVE_IMAGES_EACH"] = args["SAVE_IMAGES_EACH"] // 10 if scale > 512 else args["SAVE_IMAGES_EACH"]

        if scale_index > 0:
            epoch_len = hyperparams["steps_per_scale"][scale]
            for key, tracker in loss_stats.items():
                tracker.expand_buffer(block_size=epoch_len)

        tracked_images = 0
        total_images_phase = args["TOTAL_IMAGES"] * epochs
        limit = int(0.5 * total_images_phase)
        alpha = utils.find_alpha(tracked_images, limit)    

        n_blocks = int(log2(scale) - 1)

        if args["CONTRAST_ENABLE"]:
            loss_dict["nce"] = []
            for idx, (_, batch_size_temp)  in enumerate(hyperparams["bs_per_scale"].items()):
                loss_dict["nce"].append(PatchNCELoss(hyperparams["nce_t"], batch_size=batch_size_temp).to(args["DEVICE"]))
            models["Fp"] = net.PatchSampleFeatureProjection(scale, patch_size=max(scale//8, 8), gpu_ids=[args["DEVICE"]], nc=5, use_perceptual=True, use_mlp=True)
            models["optF"] = Adam([
                {'params': models["Fp"].parameters()},
            ], lr=0.0002, betas=(0.5, 0.999))
            

        if n_blocks > args["ENABLE_DISC_PATCH"]:
            models["Dp"] = net.PatchDiscriminator(patch_size=max(scale//8, 8), 
                                                n_layers=n_blocks, 
                                                ndf=32, 
                                                no_antialias=True, 
                                                scale=scale).to(args["DEVICE"])
            models["optDp"] = Adam([
                {'params': list(models["G"].parameters()) + list(models["Dp"].parameters())},
            ], lr=lr_per_resolution[4], betas=(0.5, 0.99))
        
        
        warmup = True
        epoch_start = time()
        print(f"[INFO]\t Starting phase {scale_num} at {scale}x{scale} scale training, {total_images_phase} total images this phase - alpha becomes 1 at {limit} and output saved every {args["SAVE_IMAGES_EACH"]} batches ({int(args["SAVE_IMAGES_EACH"]*batch_size)} images)")
        
        # Set necessary learning rate
        for opt in [models["optAE"], models["optG"], models["optD"]]:
            adjust_lr(opt, lr_per_resolution[scale])
        total_batches = len(fract)//batch_size
        extra_images = len(fract)%batch_size
        with tqdm_notebook(total=epochs*total_batches*batch_size, unit='Images', unit_scale=True, unit_divisor=1, desc="Epochs") as pbar:
            dataloader = dataset.make_fractal_alae_dataloader(fract, batch_size, image_size=scale, num_workers=3)
            loss_dict["per"] = GeneralPerceptualLoss(models["D"], 4)
            for epoch in range(epochs):
                for batch_idx, real_samples in enumerate(dataloader):
                    labels = torch.Tensor(np.array(list(range(batch_idx*batch_size, (batch_idx*batch_size)+batch_size)))).to(args["DEVICE"])
                    bs = batch_size
                    ncrops = 1
                    # In the paper 500k with blending & 500k with alpha=1 for each scale
                    alpha = utils.find_alpha(tracked_images, limit)

                    # Discriminator loss
                    z1 = utils.sample_noise(batch_size, code=code_length, device=args["DEVICE"])
                    z2 = utils.sample_noise(batch_size, code=code_length, device=args["DEVICE"])                
                    w = models["F"](z1, scale, z2, p_mix=0.9)
                    
                    real_samp = real_samples.to(args["DEVICE"]).requires_grad_()
                    fake_samples = models["G"](w, scale, alpha).detach()

                    lossD = loss_dict["discriminator"](models["E"], models["D"], alpha, real_samp, fake_samples)
                                        
                    models["optD"].zero_grad()                    
                    loss_stats["D"].update(lossD)
                    models["optD"].step()

                    # Sample 'styles' from normal distibution to be mixed
                    z1 = utils.sample_noise(batch_size, code=code_length, device=args["DEVICE"])
                    z2 = utils.sample_noise(batch_size, code=code_length, device=args["DEVICE"])
                    # Project styles to latent space
                    w = models["F"](z1, scale, z2, p_mix=0.9)
                    fake_samples = models["G"](w, scale, alpha)
                    # Generator loss                
                    lossG = loss_dict["generator"](models["E"], models["D"], alpha, fake_samples)
                    
                    models["optG"].zero_grad()
                    loss_stats["G"].update(lossG)

                    z1 = utils.sample_noise(batch_size, code=code_length, device=args["DEVICE"])
                    z2 = utils.sample_noise(batch_size, code=code_length, device=args["DEVICE"])
                    w = models["F"](z1, scale, z2, p_mix=0.9).detach()

                    lossGAvg = loss_dict["avg_generator"](models["G"], models["G_average"], w, scale, alpha)

                    loss_stats["Ga"].update(lossGAvg)
                    # Sample 'styles' from normal distibution to be mixed
                    z1 = utils.sample_noise(batch_size, code=code_length, device=args["DEVICE"])
                    real_samp = real_samples.to(args["DEVICE"])
                    E_z = models["E"](real_samp, alpha)
                    # Project styles to latent space
                    w = models["F"](z1, scale, E_z, p_mix=0.75).detach()              
                    fake_samples = models["G"](w, scale, alpha)
                    real_samples_grad = real_samples.to(args["DEVICE"]).requires_grad_()

                    lossGcon = loss_dict["generator_consistency"](fake_samples, real_samples_grad, 
                                                                loss_fn=lambda x,y: piq.MSID()(x.reshape(batch_size, -1), y.reshape(batch_size, -1)), 
                                                                use_perceptual=True,
                                                                use_ssim=True,
                                                                ssim_weight=10,
                                                                use_ssim_tv=False,
                                                                use_sobel=True,
                                                                sobel_weight=0.1,
                                                                use_sobel_tv=True,
                                                                sobel_fn=nn.L1Loss())

                    loss_stats["Gc"].update(lossGcon)
                    
                    # Sample 'styles' from normal distibution to be mixed
                    z1 = utils.sample_noise(batch_size, code=code_length, device=args["DEVICE"])
                    real_samp = real_samples.to(args["DEVICE"])
                    E_z = models["E"](real_samp, alpha)
                    # Project styles to latent space
                    w = models["F"](z1, scale, E_z, p_mix=0.75).detach()              
                    fake_samples = models["G"](w, scale, alpha)                
                    rand_samples = torch.randn(batch_size, 3, scale, scale).to(args["DEVICE"]).requires_grad_()
                    lossGsanity = loss_dict["generator_consistency"](fake_samples, rand_samples, 
                                                                loss_fn=lambda x,y: loss_dict["fft"](x, y).mean(), 
                                                                use_ssim=False,
                                                                ssim_weight=1000,
                                                                use_ssim_tv=False,
                                                                use_sobel=False,
                                                                sobel_weight=1,
                                                                use_sobel_tv=False,
                                                                sobel_fn=piq.ssim)

                    loss_stats["Gs"].update(lossGsanity)

                    # Reconstruction loss to make generator more like real samples
                    try:
                        real_samp = real_samples.to(args["DEVICE"]).requires_grad_()
                        E_z = models["E"](real_samp, alpha).repeat(batch_size, int(log2(scale)-1), 1).detach()
                        recon_samples = models["G"](E_z, scale, alpha)

                        lossGmsd = piq.MDSILoss(data_range=1.)(stand(recon_samples), stand(real_samp))
                        loss_stats["Gm"].update(lossGmsd)
                    except:
                        loss_stats["Gm"].update(losses.ssim_loss(recon_samples, real_samp))

                    real_samp = real_samples.to(args["DEVICE"]).requires_grad_()
                    E_z = models["E"](real_samp, alpha).repeat(batch_size, int(log2(scale)-1), 1).detach()
                    recon_samples = models["G"](E_z, scale, alpha)

                    lossGrec = loss_dict["generator_consistency"](recon_samples, real_samp, 
                                                                loss_fn=color_vect_loss,
                                                                use_perceptual=False,
                                                                use_ssim=True,
                                                                ssim_weight=100,
                                                                use_ssim_tv=False,
                                                                use_sobel=False,
                                                                sobel_weight=10,
                                                                use_sobel_tv=False,
                                                                sobel_fn=loss_dict["fft"])

                    loss_stats["Gr"].update(lossGrec)
                    models["optG"].step()

                    # Autoencoder loss
                    z_input = utils.sample_noise(batch_size, code=code_length, device=args["DEVICE"])

                    lossAE = loss_dict["autoencoder"](models["F"], models["G"], models["E"], scale, alpha, z_input, 
                                                    loss_fn=nn.CosineEmbeddingLoss(), 
                                                    labels=torch.cat([torch.ones([batch_size]), torch.zeros([batch_size])]).to(args["DEVICE"]))

                    models["optAE"].zero_grad()
                    loss_stats["AE"].update(lossAE)
                    models["optAE"].step()   
                

                    # Contrastive
                    if args["CONTRAST_ENABLE"]:
                        real_samp = real_samples.to(args["DEVICE"]).requires_grad_()
                        E_z = models["E"](real_samp, alpha).repeat(batch_size, int(log2(scale)-1), 1).detach()
                        recon_samples = models["G"](E_z, scale, alpha)
                        lossNCE = loss_dict["NCE_new"](models["Fp"], 
                                                    loss_dict["nce"], labels, 5, 
                                                    recon_samples, 
                                                    real_samp, 
                                                    bs * ncrops, scale, alpha)
                        models["optF"].zero_grad()
                        models["optG"].zero_grad()
                        loss_stats["NCE"].update(lossNCE)
                        models["optF"].step()
                        models["optG"].step()
                    elif args["CONTRAST_ORIG_ENABLE"]:
                        z1 = utils.sample_noise(batch_size, code=code_length, device=args["DEVICE"])
                        z2 = utils.sample_noise(batch_size, code=code_length, device=args["DEVICE"])                
                        w_tgt = models["F"](z1, scale, z2, p_mix=0.9).detach()
                        
                        E_z = models["E"](real_samp, alpha)
                        w_src = models["F"](E_z, scale, E_z, p_mix=0.9)

                        lossNCE = loss_dict["NCE_orig"](models["G"], models["Fp"], 
                                                    loss_dict["nce"], 
                                                    [n for n in range(n_blocks)], 
                                                    w_src, w_tgt, 4, scale, alpha)

                        models["optF"].zero_grad()
                        models["optG"].zero_grad()
                        loss_stats["NCE"].update(lossNCE)
                        models["optF"].step()
                        models["optG"].step()
                    

                    # Discriminator patch loss
                    if n_blocks > args["ENABLE_DISC_PATCH"]:
                        z1 = utils.sample_noise(batch_size, code=code_length, device=args["DEVICE"])
                        z2 = utils.sample_noise(batch_size, code=code_length, device=args["DEVICE"])                
                        w = models["F"](z1, scale, z2, p_mix=0.9).detach()
                        
                        real_samp = real_samples.to(args["DEVICE"]).requires_grad_()
                        fake_samples = models["G"](w, scale, alpha).detach()

                        lossD2 = loss_dict["discriminator_patch"](models["Dp"], real_samp, fake_samples, 
                                                        gamma=5, use_bce=True)
                        
                        models["optDp"].zero_grad()
                        loss_stats["Dp"].update(lossD2)
                        models["optDp"].step()

                    tracked_images += real_samples.shape[0]
                    
                    # Keep average version of Generator
                    models["G_average"].ema(models["G"], beta=0.999) 

                    # increment progress bar  
                    experiment.set_step(step)
                    step += 1
                    increment_amount = real_samples.shape[0]                
                    pbar.update(increment_amount)
                    
                    if (step % args["SAVE_IMAGES_EACH"]) == 0:
                        data = {
                            "real_samp": real_samp,
                            "real_samples": real_samples,
                            "fake_samples": fake_samples,
                            "scale": scale,
                            "alpha": alpha,
                            "field_100": set_random_field_100,
                            "field_9": set_random_field_9
                        }
                        if USE_CLR_DATA:
                            groupsize = ncrops
                        else:
                            groupsize = None
                        epoch_errors, error_msg = create_previews(models, data, step, experiment_root, epoch_errors, error_msg, groupsize=groupsize)
                        
                        # save model checkpoint
                        if step % 100 == 0:
                            make_plot_result_msg = save_model(models, experiment_root, scale, step=step, name="checkpoint")

                        pbar.set_postfix(alpha=round(alpha, 3), epoch_errors=epoch_errors, 
                            error_msg=error_msg, make_plot_result_msg=make_plot_result_msg, 
                            step=step, refresh=False)
                        
                # Save plot of loss
                #make_plot_result_msg = make_plots(loss_stats, experiment_root)
                pbar.set_postfix(alpha=round(alpha, 3), epoch_errors=epoch_errors, error_msg=error_msg, 
                                step=step, make_plot_result_msg=make_plot_result_msg, refresh=False)

                if hyperparams["use_scheduling"] & (alpha > 0.99) & (epoch in [int(epochs * 0.6), int(epochs * 0.8)]):
                    scheduler_D.step()
                    scheduler_G.step()
                    scheduler_AE.step()
            make_plot_result_msg = save_model(models, experiment_root, scale, step=step, name="final")
            pbar.set_postfix(alpha=round(alpha, 3), epoch_errors=epoch_errors, error_msg=error_msg, 
                                step=step, make_plot_result_msg=make_plot_result_msg, refresh=False) 
