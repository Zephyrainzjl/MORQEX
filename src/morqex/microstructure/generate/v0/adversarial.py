import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import matplotlib
import os

from .operations import *
from .patches import CreateTensorDataset
from .settings import ConfigProcessor
from .synthesis import *


def Trainer(config_path):
    matplotlib.use('Agg')
    configs = ConfigProcessor(config_path)

    pth = configs.cfg.output_path.out_pth
    ckpt_path = pth + '_checkpoint.pt'

    # ---------------- 数据路径与各向同性判断 ----------------
    train_data_path = configs.cfg.data.data_path
    if len(train_data_path) == 1:
        train_data_path *= 3
        isotropic = True
    else:
        isotropic = False

    print('加载用于训练的张量数据集...')
    img_channels = configs.cfg.data.img_channels
    datatype = configs.cfg.data.data_type
    imtype = configs.cfg.data.image_type
    image_size = configs.cfg.data.image_size
    scale = configs.cfg.data.scale

    tensor_dataset = CreateTensorDataset(
        train_data_path, datatype, image_size, scale
    )

    # ---------------- 模型参数 ----------------
    z_channels = configs.cfg.model_params.z_channels
    d_kernel = configs.cfg.model_params.d_kernel
    d_stride = configs.cfg.model_params.d_stride
    d_feature_channels = configs.cfg.model_params.d_feature_channels
    d_padding = configs.cfg.model_params.d_padding
    g_scale = configs.cfg.model_params.g_scale
    g_feature_channels = configs.cfg.model_params.g_feature_channels

    Discriminator, Generator = upsample_net(
        pth, True, imtype,
        d_kernel, d_stride,
        d_feature_channels, d_padding,
        g_scale, g_feature_channels
    )
    if configs.cfg.model_version == "conv":
        Discriminator, Generator = upsample_net_conv(
            pth, True, imtype,
            d_kernel, d_stride,
            d_feature_channels, d_padding,
            g_scale, g_feature_channels
        )
    
    # ---------------- 训练参数 ----------------
    num_gpu = configs.cfg.training_params.num_gpu
    num_epochs = configs.cfg.training_params.num_epochs
    batch_size = configs.cfg.training_params.batch_size
    D_batch_size = configs.cfg.training_params.Discriminator_batch_size
    lr_Generator = configs.cfg.training_params.lr_Generator
    lr_Discriminator = configs.cfg.training_params.lr_Discriminator
    beta1 = configs.cfg.training_params.beta1
    beta2 = configs.cfg.training_params.beta2
    Lambda = configs.cfg.training_params.Lambda
    critic_iters = configs.cfg.training_params.critic_iters
    cudnn.benchmark = configs.cfg.training_params.cudnn_benchmark
    workers = configs.cfg.training_params.workers
    lz = configs.cfg.training_params.lz

    cudnn.benchmark = cudnn.benchmark

    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu"
    )
    print(device, " will be used.\n")

    # ---------------- DataLoader ----------------
    dataloaderx = torch.utils.data.DataLoader(
        tensor_dataset[0], batch_size=batch_size,
        shuffle=True, num_workers=workers
    )
    dataloadery = torch.utils.data.DataLoader(
        tensor_dataset[1], batch_size=batch_size,
        shuffle=True, num_workers=workers
    )
    dataloaderz = torch.utils.data.DataLoader(
        tensor_dataset[2], batch_size=batch_size,
        shuffle=True, num_workers=workers
    )

    # ---------------- Generator ----------------
    netG = Generator().to(device)
    if ('cuda' in str(device)) and (num_gpu > 1):
        netG = nn.DataParallel(netG, list(range(num_gpu)))

    optG = optim.Adam(
        netG.parameters(), lr=lr_Generator, betas=(beta1, beta2)
    )

    # ---------------- Discriminators ----------------
    netDs = []
    optDs = []
    for i in range(3):
        netD = Discriminator()
        netD = nn.DataParallel(netD, list(range(num_gpu))).to(device)
        netDs.append(netD)
        optDs.append(
            optim.Adam(
                netD.parameters(),
                lr=lr_Discriminator,
                betas=(beta1, beta2)
            )
        )

    # ---------------- 日志变量（必须先定义） ----------------
    disc_real_log = []
    disc_fake_log = []
    Wass_log = []
    gp_log = []

    start_epoch = 0
    elapsed_time = 0.0

    # ---------------- Checkpoint Resume ----------------
    if os.path.exists(ckpt_path):
        print(f"发现 checkpoint，正在恢复：{ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        netG.load_state_dict(checkpoint['netG'])
        optG.load_state_dict(checkpoint['optG'])

        for i in range(3):
            netDs[i].load_state_dict(checkpoint['netDs'][i])
            optDs[i].load_state_dict(checkpoint['optDs'][i])

        start_epoch = checkpoint['epoch'] + 1

        disc_real_log = checkpoint['disc_real_log']
        disc_fake_log = checkpoint['disc_fake_log']
        Wass_log = checkpoint['Wass_log']
        gp_log = checkpoint['gp_log']

        elapsed_time = checkpoint.get('elapsed_time', 0.0)

        print(f"✔ 恢复至 epoch {start_epoch}")
        print(f"✔ 已训练时间 {elapsed_time / 3600:.2f} 小时")
    else:
        print("未发现 checkpoint，从头训练")

    # ---------------- 训练开始时间（关键） ----------------
    start_time = time.time() - elapsed_time

    print("Starting Training Loop...")

    # ================= Training Loop =================
    for epoch in range(start_epoch, num_epochs):
        for i, (datax, datay, dataz) in enumerate(
            zip(dataloaderx, dataloadery, dataloaderz), 1
        ):
            dataset = [datax, datay, dataz]

            noise = torch.randn(
                D_batch_size, z_channels, lz, lz, lz, device=device
            )

            generated_data = netG(noise).detach()

            # -------- Train Discriminators --------
            for dim, (netD, optimizer, data, d1, d2, d3) in enumerate(
                zip(
                    netDs, optDs, dataset,
                    [2, 3, 4], [3, 2, 2], [4, 4, 3]
                )
            ):
                if isotropic:
                    netD = netDs[0]
                    optimizer = optDs[0]

                netD.zero_grad()

                real_data = data[0].to(device)
                out_real = netD(real_data).view(-1).mean()
                # print(generated_data.shape)
                generated_data_perm = generated_data.permute(
                    0, d1, 1, d2, d3
                ).reshape(
                    image_size * D_batch_size,
                    img_channels, image_size, image_size
                )

                out_fake = netD(generated_data_perm).mean()

                gradient_penalty = calc_gradient_penalty(
                    netD,
                    real_data,
                    generated_data_perm[:batch_size],
                    batch_size,
                    image_size,
                    device,
                    Lambda,
                    img_channels
                )

                disc_cost = out_fake - out_real + gradient_penalty
                disc_cost.backward()
                optimizer.step()

            disc_real_log.append(out_real.item())
            disc_fake_log.append(out_fake.item())
            Wass_log.append(out_real.item() - out_fake.item())
            gp_log.append(gradient_penalty.item())

            # -------- Train Generator --------
            if i % int(critic_iters) == 0:
                netG.zero_grad()
                errG = 0

                noise = torch.randn(
                    batch_size, z_channels, lz, lz, lz, device=device
                )
                generated = netG(noise)

                for dim, (netD, d1, d2, d3) in enumerate(
                    zip(netDs, [2, 3, 4], [3, 2, 2], [4, 4, 3])
                ):
                    if isotropic:
                        netD = netDs[0]

                    generated_data_perm = generated.permute(
                        0, d1, 1, d2, d3
                    ).reshape(
                        image_size * batch_size,
                        img_channels, image_size, image_size
                    )

                    errG -= netD(generated_data_perm).mean()

                errG.backward()
                optG.step()

            # -------- 可视化 --------
            if i % 25 == 0:
                netG.eval()
                with torch.no_grad():
                    torch.save(netG.state_dict(), pth + '_Gen.pt')
                    torch.save(netD.state_dict(), pth + '_Disc.pt')
                    noise = torch.randn(
                        1, z_channels, lz, lz, lz, device=device
                    )
                    img = netG(noise)

                    steps = len(dataloaderx)
                    show_training_time(
                        steps, time.time(), start_time,
                        i, epoch, num_epochs
                    )

                    training_plotter(img, 3, imtype, pth)
                    dinamic_plot(
                        [disc_real_log, disc_fake_log],
                        ['Discriminator Loss', 'Generator Loss'],
                        pth, 'LossGraph'
                    )
                    dinamic_plot(
                        [Wass_log], ['Wass Distance'],
                        pth, 'WassGraph'
                    )
                    dinamic_plot(
                        [gp_log], ['Gradient Penalty'],
                        pth, 'GpGraph'
                    )
                netG.train()

        # -------- Epoch-level Checkpoint --------
        checkpoint = {
            'epoch': epoch,
            'netG': netG.state_dict(),
            'optG': optG.state_dict(),
            'netDs': [netDs[i].state_dict() for i in range(3)],
            'optDs': [optDs[i].state_dict() for i in range(3)],

            'disc_real_log': disc_real_log,
            'disc_fake_log': disc_fake_log,
            'Wass_log': Wass_log,
            'gp_log': gp_log,

            'elapsed_time': time.time() - start_time
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved at epoch {epoch}")
