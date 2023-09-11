import os
import gc
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.utils.data
import torch.nn.functional as F
import random
import argparse
import wandb
from parameters import *
from utils import *
from MDN_loss import NLL_loss, EVOLV_NLL_loss
from net import GRU_ATT
from data_loader import dataloader
from tqdm import tqdm

gc.collect()
torch.cuda.empty_cache()
random_seed = 11
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


def train(device, save_path, learning_rate, hidden_size, batch_size, input_len,
          output_len, input_size, output_size, num_layers, epochs, weight_decay, weight_pix, weight_angle,
          dropout, mode, temp):
    net = GRU_ATT(hidden_size=hidden_size, input_size=input_size,
                  output_size=output_size, output_len=output_len,
                  num_layers=num_layers, mode=mode, temp=temp, dropout=dropout).to(device=device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=1e-4, T_0=11)
    mse_loss = torch.nn.MSELoss(reduction='sum')

    train_dataset = dataloader('../downsample_data/train_align_data_1',
                               '../downsample_data/train_align_data_2')
    val_dataset = dataloader('../downsample_data/val_align_data_1',
                             '../downsample_data/val_align_data_2')

    val_size = len(val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(val_size / 2), num_workers=8)
    batch_num = len(train_loader)

    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)

    max_val_loss = float('inf')

    train_loss_list = []
    val_loss_list = []

    wandb.init(
        project="new_baseline",
        name='original attention',
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "weight_pix": weight_pix,
            "weight_angle": weight_angle,
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "temp": temp
        }
    )

    for epoch in tqdm(range(epochs)):
        net.train()
        train_loss = 0
        winner_world_train_loss = 0
        pix_1_train_loss = 0
        pix_2_train_loss = 0
        angle_1_train_loss = 0
        angle_2_train_loss = 0
        NLL_train_loss = 0
        EVOLV_NLL_train_loss = 0
        mode_train_weight = torch.zeros([output_len, mode], device=device)

        for history_rot_pix, future_diff_world, history_pos_pix_1, history_pos_pix_2, future_pos_pix_1, future_pos_pix_2, future_pos_pix_undist_1, \
                future_pos_pix_undist_2, history_pos_world in train_loader:
            optimizer.zero_grad()
            future_diff_world = future_diff_world.to(device=device)
            future_pos_pix_1 = future_pos_pix_1.to(device=device)
            future_pos_pix_2 = future_pos_pix_2.to(device=device)
            history_pos_pix_1 = history_pos_pix_1.to(device=device)
            history_pos_pix_2 = history_pos_pix_2.to(device=device)

            noise_1 = torch.concat(
                (torch.clip(torch.normal(mean=0, std=3, size=[history_pos_pix_1.shape[0], input_len, 1],
                                         device=device), -15, 15) / 848,
                 torch.clip(torch.normal(mean=0, std=3, size=[history_pos_pix_1.shape[0], input_len, 1],
                                         device=device), -15, 15) / 800), dim=2)
            noise_2 = torch.concat(
                (torch.clip(torch.normal(mean=0, std=3, size=[history_pos_pix_2.shape[0], input_len, 1],
                                         device=device), -15, 15) / 848,
                 torch.clip(torch.normal(mean=0, std=3, size=[history_pos_pix_2.shape[0], input_len, 1],
                                         device=device), -15, 15) / 800), dim=2)
            history_pos_pix_1_noise = history_pos_pix_1 + noise_1
            history_pos_pix_2_noise = history_pos_pix_2 + noise_2

            last_pos_world = history_pos_world[:, -1:, :].to(device=device)
            future_diff_world_sum = torch.cumsum(future_diff_world, dim=1)
            future_pos_world = future_diff_world_sum + last_pos_world
            history_pos_pix = torch.concat((history_pos_pix_1_noise, history_pos_pix_2_noise), dim=2)
            pred_last_pos_world_mean, pred_last_pos_world_var, pi, mu, var = net(history_pos_pix, device)
            # pi = F.softmax(pi / temp, dim=-1)
            mode_train_weight += torch.sum(pi, dim=0) / batch_size

            # pred_last_pos_world_mean_expand = torch.tile(pred_last_pos_world_mean.unsqueeze(2),
            #                                              dims=(1, 1, mode, 1))
            # pred_last_pos_world_var_expand = torch.tile(pred_last_pos_world_var.unsqueeze(2), dims=(1, 1, mode, 1))
            pred_pos_world_mean_expand = pred_last_pos_world_mean + torch.cumsum(mu, dim=1)
            pred_pos_world_var_expand = pred_last_pos_world_var + torch.cumsum(var, dim=1)

            future_pos_world_expand = torch.tile(future_pos_world.unsqueeze(2), dims=(1, 1, mode, 1))
            distance = torch.sqrt(torch.sum((pred_pos_world_mean_expand - future_pos_world_expand) ** 2, dim=-1))
            min_distance, _ = torch.min(distance, dim=-1)
            world_pos_loss = torch.sum(min_distance ** 2) / (batch_size * output_len)
            winner_world_train_loss += world_pos_loss.item()

            if epoch < 400:
                pick_num = 2 ** ((399 - epoch) // 100)
                _, index_pi = torch.topk(distance, k=pick_num, dim=2, largest=False)
                index_mu_var_world = torch.tile(index_pi.unsqueeze(-1), dims=(1, 1, 1, output_size))
                index_mu_var_image = torch.tile(index_pi.unsqueeze(-1), dims=(1, 1, 1, 2))

                pred_pos_world_mean_select = torch.gather(pred_pos_world_mean_expand, dim=2,
                                                          index=index_mu_var_world)
                pred_pos_world_var_select = torch.gather(pred_pos_world_var_expand, dim=2, index=index_mu_var_world)
                future_pos_world_select = torch.tile(future_pos_world.unsqueeze(2), dims=(1, 1, pick_num, 1))

                evolv_nll_loss = EVOLV_NLL_loss(pred_pos_world_mean_select, pred_pos_world_var_select,
                                                future_pos_world_select)
                EVOLV_NLL_train_loss += evolv_nll_loss.item()

                pred_pos_world_flat = pred_pos_world_mean_expand.reshape(batch_size, -1, 3)
                future_pos_world_flat = future_pos_world_expand.reshape(batch_size, -1, 3)

                ## calcualte future pixel position
                future_pos_pix_dist_unnorm_1 = (future_pos_pix_1 + 1) / 2 * normalize.t().repeat(batch_size,
                                                                                                 output_len,
                                                                                                 1)
                future_pos_pix_dist_unnorm_2 = (future_pos_pix_2 + 1) / 2 * normalize.t().repeat(batch_size,
                                                                                                 output_len,
                                                                                                 1)

                future_pos_pix_dist_unnorm_1 = torch.tile(future_pos_pix_dist_unnorm_1.unsqueeze(2),
                                                          dims=(1, 1, mode, 1))
                future_pos_pix_dist_unnorm_2 = torch.tile(future_pos_pix_dist_unnorm_2.unsqueeze(2),
                                                          dims=(1, 1, mode, 1))

                future_pos_pix_dist_unnorm_1 = torch.gather(future_pos_pix_dist_unnorm_1, dim=2,
                                                            index=index_mu_var_image)
                future_pos_pix_dist_unnorm_2 = torch.gather(future_pos_pix_dist_unnorm_2, dim=2,
                                                            index=index_mu_var_image)

                ## calculate predicted pixel position
                pred_pos_pix_dist_unnorm_1 = projection(
                    torch.concat(
                        (pred_pos_world_flat, torch.ones((batch_size, output_len * mode, 1), device=device)),
                        dim=2).permute(0, 2, 1),
                    extrinsic_1, intrinsic, normalize, optical_center, focal_length, batch_size, K)
                pred_pos_pix_dist_unnorm_2 = projection(
                    torch.concat(
                        (pred_pos_world_flat, torch.ones((batch_size, output_len * mode, 1), device=device)),
                        dim=2).permute(0, 2, 1),
                    extrinsic_2, intrinsic, normalize, optical_center, focal_length, batch_size, K)

                pred_pos_pix_dist_unnorm_1 = pred_pos_pix_dist_unnorm_1.reshape(batch_size, output_len, mode, -1)
                pred_pos_pix_dist_unnorm_2 = pred_pos_pix_dist_unnorm_2.reshape(batch_size, output_len, mode, -1)

                pred_pos_pix_dist_unnorm_1 = torch.gather(pred_pos_pix_dist_unnorm_1, dim=2,
                                                          index=index_mu_var_image)
                pred_pos_pix_dist_unnorm_2 = torch.gather(pred_pos_pix_dist_unnorm_2, dim=2,
                                                          index=index_mu_var_image)

                pix_1_loss = torch.nanmean(
                    torch.sum(torch.abs(pred_pos_pix_dist_unnorm_1 - future_pos_pix_dist_unnorm_1), dim=-1))
                pix_2_loss = torch.nanmean(
                    torch.sum(torch.abs(pred_pos_pix_dist_unnorm_2 - future_pos_pix_dist_unnorm_2), dim=-1))

                pix_1_train_loss += pix_1_loss.item()
                pix_2_train_loss += pix_2_loss.item()

                ## calculate angle
                angle_left, angle_right = calculate_angle(pred_pos_world_flat, future_pos_world_flat, left_cam_pos,
                                                          right_cam_pos)
                angle_left = angle_left.reshape(batch_size, output_len, 8)
                angle_right = angle_right.reshape(batch_size, output_len, 8)
                angle_left = torch.gather(angle_left, dim=2, index=index_pi)
                angle_right = torch.gather(angle_right, dim=2, index=index_pi)
                angle_1_loss = torch.nanmean(angle_left)
                angle_2_loss = torch.nanmean(angle_right)

                angle_1_train_loss += angle_1_loss.item()
                angle_2_train_loss += angle_2_loss.item()

                loss = evolv_nll_loss + (pix_1_loss + pix_2_loss) * weight_pix + (
                        angle_1_loss + angle_2_loss) * weight_angle

            else:
                nll_loss = NLL_loss(pi, pred_pos_world_mean_expand, pred_pos_world_var_expand,
                                    future_pos_world_expand)
                NLL_train_loss += nll_loss.item()
                loss = nll_loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        mode_train_weight = mode_train_weight / batch_num
        wandb.log({'Training loss': train_loss / batch_num}, step=epoch)
        wandb.log({'Pix 1 Training loss': pix_1_train_loss / batch_num}, step=epoch)
        wandb.log({'Pix 2 Training loss': pix_2_train_loss / batch_num}, step=epoch)
        wandb.log({'Angle 1 Training loss': angle_1_train_loss / batch_num}, step=epoch)
        wandb.log({'Angle 2 Training loss': angle_2_train_loss / batch_num}, step=epoch)
        wandb.log({'Winner World Training loss': winner_world_train_loss / batch_num}, step=epoch)
        wandb.log({'NLL Training loss': NLL_train_loss / batch_num}, step=epoch)
        wandb.log({'EVOLV NLL Training loss': EVOLV_NLL_train_loss / batch_num}, step=epoch)

        # for i in range(output_len):
        #     wandb.log({'Training Step ' + str(i + 1): {'1': mode_train_weight[i, 0].item(),
        #                                                '2': mode_train_weight[i, 1].item(),
        #                                                '3': mode_train_weight[i, 2].item(),
        #                                                '4': mode_train_weight[i, 3].item(),
        #                                                '5': mode_train_weight[i, 4].item(),
        #                                                '6': mode_train_weight[i, 5].item(),
        #                                                '7': mode_train_weight[i, 6].item(),
        #                                                '8': mode_train_weight[i, 7].item()}}, step=epoch)
        train_loss_list.append(train_loss / batch_num)
        #
        winner_world_val_loss = 0
        true_winner_world_val_loss = 0
        NLL_val_loss = 0
        EVOLV_NLL_val_loss = 0
        mode_val_weight = torch.zeros([output_len, mode], device=device)
        net.eval()
        with torch.no_grad():
            for history_rot_pix, future_diff_world, history_pos_pix_1, history_pos_pix_2, future_pos_pix_1, future_pos_pix_2, future_pos_pix_undist_1, \
                    future_pos_pix_undist_2, history_pos_world in val_loader:
                future_diff_world = future_diff_world.to(device=device)
                history_pos_pix_1 = history_pos_pix_1.to(device=device)
                history_pos_pix_2 = history_pos_pix_2.to(device=device)

                noise_1 = torch.concat(
                    (torch.clip(torch.normal(mean=0, std=3, size=[history_pos_pix_1.shape[0], input_len, 1],
                                             device=device), -15, 15) / 848,
                     torch.clip(torch.normal(mean=0, std=3, size=[history_pos_pix_1.shape[0], input_len, 1],
                                             device=device), -15, 15) / 800), dim=2)
                noise_2 = torch.concat(
                    (torch.clip(torch.normal(mean=0, std=3, size=[history_pos_pix_2.shape[0], input_len, 1],
                                             device=device), -15, 15) / 848,
                     torch.clip(torch.normal(mean=0, std=3, size=[history_pos_pix_2.shape[0], input_len, 1],
                                             device=device), -15, 15) / 800), dim=2)
                history_pos_pix_1_noise = history_pos_pix_1 + noise_1
                history_pos_pix_2_noise = history_pos_pix_2 + noise_2

                last_pos_world = history_pos_world[:, -1:, :].to(device=device)
                future_diff_world_sum = torch.cumsum(future_diff_world, dim=1)
                future_pos_world = future_diff_world_sum + last_pos_world
                history_pos_pix = torch.concat((history_pos_pix_1_noise, history_pos_pix_2_noise), dim=2)
                pred_last_pos_world_mean, pred_last_pos_world_var, pi, mu, var = net(history_pos_pix, device)
                # pi = F.softmax(pi / temp, dim=-1)
                mode_val_weight += torch.sum(pi, dim=0) / val_size * 2
                # pred_last_pos_world_mean_expand = torch.tile(pred_last_pos_world_mean.unsqueeze(2),
                #                                              dims=(1, 1, mode, 1))
                # pred_last_pos_world_var_expand = torch.tile(pred_last_pos_world_var.unsqueeze(2),
                #                                             dims=(1, 1, mode, 1))
                pred_pos_world_mean_expand = pred_last_pos_world_mean + torch.cumsum(mu, dim=1)
                pred_pos_world_var_expand = pred_last_pos_world_var + torch.cumsum(var, dim=1)

                future_pos_world_expand = torch.tile(future_pos_world.unsqueeze(2), dims=(1, 1, mode, 1))
                distance = torch.sqrt(
                    torch.sum((pred_pos_world_mean_expand - future_pos_world_expand) ** 2, dim=-1))

                if epoch < 400:
                    pick_num = 2 ** ((399 - epoch) // 100)
                    _, index_pi = torch.topk(pi, k=pick_num, dim=2, largest=True)
                    index_mu_var_world = torch.tile(index_pi.unsqueeze(-1), dims=(1, 1, 1, output_size))
                    pred_pos_world_mean_select = torch.gather(pred_pos_world_mean_expand, dim=2,
                                                              index=index_mu_var_world)
                    pred_pos_world_var_select = torch.gather(pred_pos_world_var_expand, dim=2,
                                                             index=index_mu_var_world)
                    future_pos_world_select = torch.tile(future_pos_world.unsqueeze(2), dims=(1, 1, pick_num, 1))

                    evolv_nll_loss = EVOLV_NLL_loss(pred_pos_world_mean_select, pred_pos_world_var_select,
                                                    future_pos_world_select)

                    loss = evolv_nll_loss
                    EVOLV_NLL_val_loss += loss.item()
                else:
                    loss = NLL_loss(pi, pred_pos_world_mean_expand, pred_pos_world_var_expand,
                                    future_pos_world_expand)
                    NLL_val_loss += loss.item()

                index_pi = torch.argmax(pi, dim=2, keepdim=True)
                index_mu_var = torch.tile(index_pi.unsqueeze(-1), dims=(1, 1, 1, output_size))
                selected_mu = torch.gather(mu, dim=2, index=index_mu_var).squeeze()
                pred_pos_world = torch.cumsum(selected_mu, dim=1) + pred_last_pos_world_mean

                min_distance, _ = torch.min(distance, dim=-1)
                world_pos_loss = mse_loss(pred_pos_world, future_pos_world) / (val_size * output_len / 2)
                world_pos_loss_true = torch.sum(min_distance ** 2) / (val_size * output_len / 2)
                winner_world_val_loss += world_pos_loss.item()
                true_winner_world_val_loss += world_pos_loss_true.item()

            mode_val_weight = mode_val_weight / 2
            wandb.log({'NLL Validation loss': NLL_val_loss / 2}, step=epoch)
            wandb.log({'EVOLV NLL Validation loss': EVOLV_NLL_val_loss / 2}, step=epoch)
            wandb.log({'Winner World Validation loss': winner_world_val_loss / 2}, step=epoch)
            wandb.log({'True Winner World Validation loss': true_winner_world_val_loss / 2}, step=epoch)
            # for i in range(output_len):
            #     wandb.log({'Validation Step ' + str(i + 1): {'1': mode_val_weight[i, 0].item(),
            #                                                  '2': mode_val_weight[i, 1].item(),
            #                                                  '3': mode_val_weight[i, 2].item(),
            #                                                  '4': mode_val_weight[i, 3].item(),
            #                                                  '5': mode_val_weight[i, 4].item(),
            #                                                  '6': mode_val_weight[i, 5].item(),
            #                                                  '7': mode_val_weight[i, 6].item(),
            #                                                  '8': mode_val_weight[i, 7].item()}},
            #               step=epoch)
            if epoch >= 400:
                if NLL_val_loss / 2 < max_val_loss:
                    max_val_loss = NLL_val_loss / 2
                    torch.save(net.state_dict(),
                               os.path.join(save_path, 'best_model_att.pt'))

                    print("best val loss: {}".format(NLL_val_loss / 2))
            else:
                if epoch % 100 == 99:
                    torch.save(net.state_dict(),
                               os.path.join(save_path, 'model_' + str(epoch // 100 + 1) + '.pt'))
        ## early stop
        if epoch < 400:
            scheduler.step()

        if epoch == 400:
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-4,
                                         weight_decay=weight_decay)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_len', type=int, default=8)
    parser.add_argument('--output_len', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--top', type=int, default=1)
    parser.add_argument('--modes', type=int, default=8)
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--weight_pix', type=float, default=0.01)
    parser.add_argument('--weight_angle', type=float, default=10)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--temp', type=float, default=1)
    args = parser.parse_args()

    save_path = './result'
    train(device=device, save_path=save_path, learning_rate=args.lr,
          hidden_size=args.hidden_size, batch_size=args.batch, num_layers=1,
          input_len=args.input_len,
          output_len=args.output_len, input_size=4,
          output_size=3, epochs=args.epochs, weight_decay=args.weight_decay,
          weight_pix=args.weight_pix, weight_angle=args.weight_angle, dropout=args.dropout, mode=args.modes,
          temp=args.temp)
