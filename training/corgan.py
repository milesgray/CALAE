if opt.training:

    if opt.resume:
        #####################################
        #### Load model and optimizer #######
        #####################################

        # Loading the checkpoint
        checkpoint = torch.load(os.path.join(opt.PATH, "model_epoch_1000.pth"))

        # Load models
        generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
        discriminatorModel.load_state_dict(checkpoint['Discriminator_state_dict'])
        autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])
        autoencoderDecoder.load_state_dict(checkpoint['Autoencoder_Decoder_state_dict'])

        # Load optimizers
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        optimizer_A.load_state_dict(checkpoint['optimizer_A_state_dict'])

        # Load losses
        g_loss = checkpoint['g_loss']
        d_loss = checkpoint['d_loss']
        a_loss = checkpoint['a_loss']

        # Load epoch number
        epoch = checkpoint['epoch']

        generatorModel.eval()
        discriminatorModel.eval()
        autoencoderModel.eval()
        autoencoderDecoder.eval()

    for epoch_pre in range(opt.n_epochs_pretrain):
        for i, samples in enumerate(dataloader_train):

            # Configure input
            real_samples = Variable(samples.type(Tensor))

            # Generate a batch of images
            recons_samples = autoencoderModel(real_samples)

            # Loss measures generator's ability to fool the discriminator
            a_loss = autoencoder_loss(recons_samples, real_samples)

            # # Reset gradients (if you comment below line, it would be a mess. Think why?!!!!!!!!!)
            optimizer_A.zero_grad()

            a_loss.backward()
            optimizer_A.step()

            batches_done = epoch_pre * len(dataloader_train) + i
            if batches_done % opt.sample_interval == 0:
                print(
                    "[Epoch %d/%d of pretraining] [Batch %d/%d] [A loss: %.3f]"
                    % (epoch_pre + 1, opt.n_epochs_pretrain, i, len(dataloader_train), a_loss.item())
                    , flush=True)

    gen_iterations = 0
    for epoch in range(opt.n_epochs):
        epoch_start = time.time()
        for i, samples in enumerate(dataloader_train):

            # Adversarial ground truths
            valid = Variable(Tensor(samples.shape[0]).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(samples.shape[0]).fill_(0.0), requires_grad=False)

            # Configure input
            real_samples = Variable(samples.type(Tensor))

            # Sample noise as generator input
            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            for p in discriminatorModel.parameters():  # reset requires_grad
                p.requires_grad = True

            # train the discriminator n_iter_D times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                n_iter_D = 100
            else:
                n_iter_D = opt.n_iter_D
            j = 0
            while j < n_iter_D:
                j += 1

                # clamp parameters to a cube
                for p in discriminatorModel.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                # reset gradients of discriminator
                optimizer_D.zero_grad()

                errD_real = torch.mean(discriminatorModel(real_samples), dim=0)
                errD_real.backward(one)

                # Measure discriminator's ability to classify real from generated samples
                # The detach() method constructs a new view on a tensor which is declared
                # not to need gradients, i.e., it is to be excluded from further tracking of
                # operations, and therefore the subgraph involving this view is not recorded.
                # Refer to http://www.bnikolic.co.uk/blog/pytorch-detach.html.

                # Sample noise as generator input
                z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

                # Generate a batch of images
                fake_samples = generatorModel(z)

                # uncomment if there is no autoencoder
                fake_samples = torch.squeeze(autoencoderDecoder(fake_samples.unsqueeze(dim=2)))

                errD_fake = torch.mean(discriminatorModel(fake_samples.detach()),dim=0)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake

                # Optimizer step
                optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            # We’re supposed to clear the gradients each iteration before calling loss.backward() and optimizer.step().
            #
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch
            # accumulates the gradients on subsequent backward passes. This is convenient while training RNNs. So,
            # the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.
            #
            # Because of this, when you start your training loop, ideally you should zero out the gradients so
            # that you do the parameter update correctly. Else the gradient would point in some other direction
            # than the intended direction towards the minimum (or maximum, in case of maximization objectives).

            # Since the backward() function accumulates gradients, and you don’t want to mix up gradients between
            # minibatches, you have to zero them out at the start of a new minibatch. This is exactly like how a general
            # (additive) accumulator variable is initialized to 0 in code.

            for p in discriminatorModel.parameters():  # reset requires_grad
                p.requires_grad = False

            # Zero grads
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

            # Generate a batch of images
            fake_samples = generatorModel(z)

            # uncomment if there is no autoencoder
            fake_samples = torch.squeeze(autoencoderDecoder(fake_samples.unsqueeze(dim=2)))

            # Loss measures generator's ability to fool the discriminator
            errG = torch.mean(discriminatorModel(fake_samples), dim=0)
            errG.backward(one)

            # read more at https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/4
            optimizer_G.step()
            gen_iterations += 1

        with torch.no_grad():

            # Variables
            real_samples_test = next(iter(dataloader_test))
            real_samples_test = Variable(real_samples_test.type(Tensor))
            z = torch.randn(samples.shape[0], opt.latent_dim, device=device)

            # Generator
            fake_samples_test_temp = generatorModel(z)
            fake_samples_test = torch.squeeze(autoencoderDecoder(fake_samples_test_temp.unsqueeze(dim=2)))

            # Discriminator
            # F.sigmoid() is needed as the discriminator outputs are logits without any sigmoid.
            out_real_test = discriminatorModel(real_samples_test).view(-1)
            accuracy_real_test = discriminator_accuracy(F.sigmoid(out_real_test), valid)

            out_fake_test = discriminatorModel(fake_samples_test.detach()).view(-1)
            accuracy_fake_test = discriminator_accuracy(F.sigmoid(out_fake_test), fake)

            # Test autoencoder
            reconst_samples_test = autoencoderModel(real_samples_test)
            a_loss_test = autoencoder_loss(reconst_samples_test, real_samples_test)

        print('TRAIN: [Epoch %d/%d] [Batch %d/%d] Loss_D: %.3f Loss_G: %.3f Loss_D_real: %.3f Loss_D_fake %.3f'
              % (epoch + 1, opt.n_epochs, i, len(dataloader_train),
                 errD.item(), errG.item(), errD_real.item(), errD_fake.item()), flush=True)

        print(
            "TEST: [Epoch %d/%d] [Batch %d/%d] [A loss: %.2f] [real accuracy: %.2f] [fake accuracy: %.2f]"
            % (epoch + 1, opt.n_epochs, i, len(dataloader_train),
               a_loss_test.item(), accuracy_real_test,
               accuracy_fake_test)
            , flush=True)

        # End of epoch
        epoch_end = time.time()
        if opt.epoch_time_show:
            print("It has been {0} seconds for this epoch".format(epoch_end - epoch_start), flush=True)

        if (epoch + 1) % opt.epoch_save_model_freq == 0:
            # Refer to https://pytorch.org/tutorials/beginner/saving_loading_models.html
            torch.save({
                'epoch': epoch + 1,
                'Generator_state_dict': generatorModel.state_dict(),
                'Discriminator_state_dict': discriminatorModel.state_dict(),
                'Autoencoder_state_dict': autoencoderModel.state_dict(),
                'Autoencoder_Decoder_state_dict': autoencoderDecoder.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'optimizer_A_state_dict': optimizer_A.state_dict(),
            }, os.path.join(opt.expPATH, "model_epoch_%d.pth" % (epoch + 1)))

            # keep only the most recent 10 saved models
            # ls -d -1tr /home/sina/experiments/pytorch/model/* | head -n -10 | xargs -d '\n' rm -f
            call("ls -d -1tr " + opt.expPATH + "/*" + " | head -n -10 | xargs -d '\n' rm -f", shell=True)

if opt.finetuning:

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(opt.PATH, "model_epoch_100.pth"))

    # Setup model
    generatorModel = Generator()
    discriminatorModel = Discriminator()

    if cuda:
        generatorModel.cuda()
        discriminatorModel.cuda()
        discriminator_loss.cuda()

    # Setup optimizers
    optimizer_G = torch.optim.Adam(generatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Load models
    generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
    discriminatorModel.load_state_dict(checkpoint['Discriminator_state_dict'])

    # Load optimizers
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    # Load losses
    g_loss = checkpoint['g_loss']
    d_loss = checkpoint['d_loss']

    # Load epoch number
    epoch = checkpoint['epoch']

    generatorModel.eval()
    discriminatorModel.eval()

if opt.generate:

    # Check cuda
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device BUT it is not in use...")

    # Activate CUDA
    device = torch.device("cuda:0" if opt.cuda else "cpu")

    #####################################
    #### Load model and optimizer #######
    #####################################

    # Loading the checkpoint
    checkpoint = torch.load(os.path.join(opt.expPATH, "model_epoch_100.pth"))

    # Load models
    generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
    autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])
    autoencoderDecoder.load_state_dict(checkpoint['Autoencoder_Decoder_state_dict'])

    # insert weights [required]
    generatorModel.eval()
    autoencoderModel.eval()
    autoencoderDecoder.eval()

    #######################################################
    #### Load real data and generate synthetic data #######
    #######################################################

    # Load real data
    real_samples = dataset_train_object.return_data()
    num_fake_samples = 10000

    # Generate a batch of samples
    gen_samples = np.zeros_like(real_samples, dtype=type(real_samples))
    n_batches = int(num_fake_samples / opt.batch_size)
    for i in range(n_batches):
        # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))
        z = torch.randn(opt.batch_size, opt.latent_dim, device=device)
        gen_samples_tensor = generatorModel(z)
        gen_samples_decoded = torch.squeeze(autoencoderDecoder(gen_samples_tensor.unsqueeze(dim=2)))
        gen_samples[i * opt.batch_size:(i + 1) * opt.batch_size, :] = gen_samples_decoded.cpu().data.numpy()
        # Check to see if there is any nan
        assert (gen_samples[i, :] != gen_samples[i, :]).any() == False

    gen_samples = np.delete(gen_samples, np.s_[(i + 1) * opt.batch_size:], 0)
    gen_samples[gen_samples >= 0.5] = 1.0
    gen_samples[gen_samples < 0.5] = 0.0

    # Trasnform Object array to float
    gen_samples = gen_samples.astype(np.float32)

    # ave synthetic data
    np.save(os.path.join(opt.expPATH, "synthetic.npy"), gen_samples, allow_pickle=False)

def evaluate(dataset_train_object, opt, plot=False):
    # Load synthetic data
    gen_samples = np.load(os.path.join(opt.expPATH, "synthetic.npy"), allow_pickle=False)

    # Load real data
    real_samples = dataset_train_object.return_data()[0:gen_samples.shape[0], :]

    # Dimenstion wise probability
    prob_real = np.mean(real_samples, axis=0)
    prob_syn = np.mean(gen_samples, axis=0)

    if plot:
        p1 = plt.scatter(prob_real, prob_syn, c="b", alpha=0.5, label="WGAN")
        x_max = max(np.max(prob_real), np.max(prob_syn))
        x = np.linspace(0, x_max + 0.1, 1000)
        
        p2 = plt.plot(x, x, linestyle='-', color='k', label="Ideal")  # solid
        plt.tick_params(labelsize=12)
        plt.legend(loc=2, prop={'size': 15})
        # plt.title('Scatter plot p')
        # plt.xlabel('x')
        # plt.ylabel('y')
        plt.show()
