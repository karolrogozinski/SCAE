import torch
import numpy as np

import matplotlib.pyplot as plt


def plot_epoch(data_loader, model, device, taus, epoch, time):
    print("===============================================================")
    print("Completed Epoch", epoch, " Time: ", time)
    print("===============================================================")

    for (data, label) in data_loader:
        data = data.to(device)
        cond = label[:, 9:].to(device)

        reconstruct, _, z, w, _, _, w_mask = model(data, cond, taus)

        fig, axs = plt.subplots(4, 7, figsize=(15, 6))
        for i in range(28):
            if i < 7:
                x = data.cpu().detach().numpy()[i][0]
            elif i < 14:
                x = reconstruct.cpu().detach().numpy()[i-7][0]
            elif i < 21:
                reconstruct, _ = model.generate(w_mask, w=w[i-14].unsqueeze(0))
                x = reconstruct.cpu().detach().numpy()[0][0]
            else:
                reconstruct, _ = model.generate(w_mask, z=z[i-21].unsqueeze(0))
                x = reconstruct.cpu().detach().numpy()[0][0]

            im = axs[i//7, i % 7].imshow(x, interpolation='none',
                                         cmap='gnuplot')
            axs[i//7, i % 7].axis('off')
            fig.colorbar(im, ax=axs[i//7, i % 7])
        plt.show()
        break


def save_model(model, optimizer, results_dir, epoch):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f'{results_dir}modelCorrVAE_{epoch}.pt')


def sum_channels_parallel(data):
    coords = np.ogrid[0:data.shape[1], 0:data.shape[2]]
    half_x = data.shape[1]//2
    half_y = data.shape[2]//2

    checkerboard = (coords[0] + coords[1]) % 2 != 0
    checkerboard.reshape(-1,checkerboard.shape[0], checkerboard.shape[1])

    ch5 = (data*checkerboard).sum(axis=1).sum(axis=1)

    checkerboard = (coords[0] + coords[1]) % 2 == 0
    checkerboard = checkerboard.reshape(
        -1, checkerboard.shape[0], checkerboard.shape[1])

    mask = np.zeros((1, data.shape[1], data.shape[2]))
    mask[:, :half_x, :half_y] = checkerboard[:, :half_x, :half_y]
    ch1 = (data*mask).sum(axis=1).sum(axis=1)

    mask = np.zeros((1, data.shape[1], data.shape[2]))
    mask[:, :half_x, half_y:] = checkerboard[:, :half_x, half_y:]
    ch2 = (data*mask).sum(axis=1).sum(axis=1)

    mask = np.zeros((1, data.shape[1], data.shape[2]))
    mask[:, half_x:, :half_y] = checkerboard[:, half_x:, :half_y]
    ch3 = (data*mask).sum(axis=1).sum(axis=1)

    mask = np.zeros((1, data.shape[1], data.shape[2]))
    mask[:, half_x:, half_y:] = checkerboard[:, half_x:, half_y:]
    ch4 = (data*mask).sum(axis=1).sum(axis=1)

    return zip(ch1, ch2, ch3, ch4, ch5)
