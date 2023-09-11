import torch


## pi NxLxMx1
## mu, var NxLxMx3

def NLL_loss(pi, pred_pos_world_mean_expand, pred_pos_world_var_expand, future_pos_world_expand):  ## here mu is difference
    log_prob = -(torch.sum(torch.log(pred_pos_world_var_expand), dim=-1) + torch.sum(
        (pred_pos_world_mean_expand - future_pos_world_expand) ** 2 / pred_pos_world_var_expand, dim=-1)) / 2  ## NxLxM
    log_pi = torch.log(pi+1e-6)  ## NxLxM

    loss = torch.nanmean(-torch.logsumexp(log_prob + log_pi, dim=-1))

    return loss


def WTA_NLL_loss(mu, var, pred_last_pos_world, pred_last_pos_world_var, future_pos_world, index_mu_var):
    mu = torch.cumsum(mu, dim=1)
    var = torch.cumsum(var, dim=1)
    # pi = torch.gather(pi, dim=2, index=index_pi)
    mu = torch.gather(mu, dim=2, index=index_mu_var).squeeze()
    var = torch.gather(var, dim=2, index=index_mu_var).squeeze()
    var = pred_last_pos_world_var + var
    log_prob = -(torch.sum(torch.log(var), dim=-1, keepdim=True) + torch.sum(
        (mu + pred_last_pos_world - future_pos_world) ** 2 / var, dim=-1, keepdim=True)) / 2  ## NxLx1
    # log_pi = torch.log(pi) ## NxLx1

    loss = torch.nanmean(-torch.logsumexp(log_prob, dim=-1))

    return loss
def EVOLV_NLL_loss(pred_pos_world_mean_select, pred_pos_world_var_select, future_pos_world_select):
    log_prob = -(torch.sum(torch.log(pred_pos_world_var_select), dim=-1, keepdim=True)
                 + torch.sum((pred_pos_world_mean_select - future_pos_world_select) ** 2 / pred_pos_world_var_select,
                             dim=-1, keepdim=True)) / 2  ## NxLx1
    # log_pi = torch.log(pi) ## NxLx1

    loss = torch.nanmean(-torch.logsumexp(log_prob, dim=-1))

    return loss
