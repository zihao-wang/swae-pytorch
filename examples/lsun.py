import argparse
import json
import os

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision.utils as vutils
from alignlib import plot_tree2D
from torchvision import datasets, transforms
from xwae.distributions import rand_cirlce2d, rand_ring2d, rand_uniform2d, randn, rand
from xwae.models.lsun import LSUNAutoencoder
from xwae.trainer import SWAEBatchTrainer, XWAEBatchTrainer
from xwae.tree_decoding import decode_tree


def main():
    # train args
    parser = argparse.ArgumentParser(description='Sliced Wasserstein Autoencoder PyTorch LSUN Example')
    parser.add_argument('--datadir', default='../.data', help='path to dataset')
    parser.add_argument('--outdir', default='output/lsun', help='directory to output images and model checkpoints')
    parser.add_argument('--img-size', type=int, default=64, metavar='S',
                        help='input image size (default: 64)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--beta1', type=float, default=0.5, metavar='B1',
                        help='Adam beta1 (default: 0.5)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help='Adam beta2 (default: 0.999)')
    parser.add_argument('--num-projections', type=int, default=500, metavar='NP',
                        help='number of projections (default: 500)')
    parser.add_argument('--weight', type=float, default=10.0, metavar='W',
                        help='weight of divergence (default: 10.0)')
    parser.add_argument('--distribution', type=str, default='normal', metavar='DIST',
                        help='Latent Distribution (default: normal)')
    parser.add_argument('--latent_dim', type=int, default=64, metavar='ES',
                        help='model embedding size (default: 64)')
    parser.add_argument('--cuda', type=int, default=-1,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=16, metavar='N',
                        help='number of dataloader workers if device is CPU (default: 16)')
    parser.add_argument('--seed', type=int, default=7, metavar='S',
                        help='random seed (default: 7)')
    parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                        help='number of batches to log training status (default: 25)')
    parser.add_argument('--log-epoch-interval', type=int, default=1, metavar='N',
                        help='number of epochs to save training artifacts (default: 1)')
    args = parser.parse_args()

    # create output directory
    imagesdir = os.path.join(args.outdir, 'images')
    chkptdir = os.path.join(args.outdir, 'models')
    os.makedirs(args.datadir, exist_ok=True)
    os.makedirs(imagesdir, exist_ok=True)
    os.makedirs(chkptdir, exist_ok=True)

    # determine device and device dep. args
    device = torch.device("cuda:{}".format(args.cuda) if args.cuda >= 0 else "cpu")
    dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda >= 0 else {'num_workers': args.num_workers, 'pin_memory': False}

    # set random seed
    torch.manual_seed(args.seed)
    if args.cuda >= 0:
        torch.cuda.manual_seed(args.seed)

    # log args
    print('batch size {}\nimage size {}\nepochs {}\nAdam: lr {} betas {}/{}\ndistribution {}\nusing device {}\nseed set to {}'.format(
        args.batch_size, args.img_size, args.epochs,
        args.lr, args.beta1, args.beta2, args.distribution,
        device.type, args.seed
    ))

    # build train and test set data loaders
    train_loader = torch.utils.data.DataLoader(
        datasets.LSUN(args.datadir, classes=['church_outdoor_train'],
                      transform=transforms.Compose([
                        transforms.Resize(args.img_size),
                        transforms.CenterCrop(args.img_size),
                        transforms.ToTensor()
                      ])),
        batch_size=args.batch_size, shuffle=True, **dataloader_kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.LSUN(args.datadir, classes=['church_outdoor_val'],
                      transform=transforms.Compose([
                        transforms.Resize(args.img_size),
                        transforms.CenterCrop(args.img_size),
                        transforms.ToTensor()
                      ])),
        batch_size=32, shuffle=False, **dataloader_kwargs)

    # create encoder and decoder
    model = LSUNAutoencoder(embedding_dim=args.embedding_size).to(device)
    # print(model)

    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # determine latent distribution
    if args.latent_dim == 2:
        if args.distribution == 'circle':
            distribution_fn = rand_cirlce2d
        elif args.distribution == 'ring':
            distribution_fn = rand_ring2d
        else:
            distribution_fn = rand_uniform2d
    else:
        distribution_fn = randn(args.latent_dim)
    # create batch sliced_wasserstein autoencoder trainer
    trainer = XWAEBatchTrainer(model, optimizer, distribution_fn,
                               align_name=args.align_name, dim=args.latent_dim,
                               fraction=args.fraction, weight=args.weight, device=device)

    # put networks in training mode
    model.train()
    # train networks for n epochs
    print('training...')
    for epoch in range(args.epochs):
        if epoch > 10:
            trainer.weight *= 1.1

        # train autoencoder on train dataset
        for batch_idx, (x, y) in enumerate(train_loader, start=0):
            batch = trainer.train_on_batch(x)
            if (batch_idx + 1) % args.log_interval == 0:
                print('Train Epoch: {} ({:.2f}%) [{}/{}]\tLoss: {:.6f}'.format(
                        epoch + 1, float(epoch + 1) / (args.epochs) * 100.,
                        (batch_idx + 1), len(train_loader),
                        batch['loss'].item()))
        fig, ax = plt.subplots(figsize=(10, 10))
        z = batch['encode'].detach().cpu().numpy()
        ax.scatter(z[:, 0], z[:, 1], c=(10 * y.cpu().numpy()), cmap=plt.cm.Spectral)
        # ax.set_xlim([-1.5, 1.5])
        # ax.set_ylim([-1.5, 1.5])
        # ax.set_title('Test Latent Space\nLoss: {:.5fx}'.format(test_loss))
        if args.align_name == 'atw':
            plot_tree2D(ax, trainer.align_method.dtms.root, 1)
        fig.savefig('{}/debug-epoch-{}.png'.format(imagesdir, epoch+1))
        plt.close()

        # evaluate autoencoder on test dataset
        test_encode, test_targets, test_loss = list(), list(), 0.0
        with torch.no_grad():
            for test_batch_idx, (x_test, y_test) in enumerate(test_loader, start=0):
                test_evals = trainer.test_on_batch(x_test)
                test_encode.append(test_evals['encode'].detach())
                test_loss += test_evals['loss'].item()
                test_targets.append(y_test)
        test_encode, test_targets = torch.cat(test_encode).cpu().numpy(), torch.cat(test_targets).cpu().numpy()
        test_loss /= len(test_loader)
        print('Test Epoch: {} ({:.2f}%)\tLoss: {:.6f}'.format(
                epoch + 1, float(epoch + 1) / (args.epochs) * 100.,
                test_loss))
        print('{{"metric": "loss", "value": {}}}'.format(test_loss))

        # save encoded samples plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(test_encode[:, 0], -test_encode[:, 1], c=(10 * test_targets), cmap=plt.cm.Spectral)
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_title('Test Latent Space\nLoss: {:.5f}'.format(test_loss))
        if args.align_name == 'atw':
            plot_tree2D(ax, trainer.align_method.dtms.root, 1)
        fig.savefig('{}/test_latent_epoch_{}.png'.format(imagesdir, epoch + 1))
        plt.close()

        # save sample input and reconstruction
        vutils.save_image(x, '{}/test_samples_epoch_{}.png'.format(imagesdir, epoch + 1))
        vutils.save_image(batch['decode'].detach(),
                          '{}/test_reconstructions_epoch_{}.png'.format(imagesdir, epoch + 1),
                          normalize=True)

        # save tree decoding
        if args.align_name == 'atw':
            decode_tree(lambda x: model.decoder(x),
                        trainer.align_method.dtms,
                        '{}/latent_tree_node_decode_epoch_{}'.format(imagesdir, epoch+1), device=device)
            latent_tree = trainer.align_method.dtms.to_dict()
            with open("{}/latent_tree_epoch_{}.json".format(imagesdir, epoch+1), 'wt') as f:
                json.dump(latent_tree, f)


if __name__ == '__main__':
    main()
