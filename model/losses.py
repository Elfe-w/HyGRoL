
import torch
import torch.nn as nn
import torch

def supcon(z, y, temperature=0.5, base_temperature=0.07):
    '''
    Supervised normalized temperature-scaled cross entropy loss.
    A variant of Multi-class N-pair Loss from (Sohn 2016)
    Later used in SimCLR (Chen et al. 2020, Khosla et al. 2020).
    Implementation modified from:
        - https://github.com/google-research/simclr/blob/master/objective.py
        - https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Args:
        z: hidden vector of shape [bsz, n_features].
        y: ground truth of shape [bsz].
    '''
    device = (torch.device('cuda')
              if z.is_cuda
              else torch.device('cpu'))

    batch_size = z.shape[0]
    contrast_count = 1
    anchor_count = contrast_count
    y = y.unsqueeze(-1)

    mask = torch.eq(y, y.T).float().to(device)
    anchor_dot_contrast = torch.div(
        torch.matmul(z, z.T),
        temperature)

    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits_min, _ = torch.min(anchor_dot_contrast, dim=1, keepdim=True)
    logits = (anchor_dot_contrast - logits_min.detach()) / (logits_max - logits_min)
    # 任意两个特征之间的结果减去最大值，为了数值稳定

    # # tile mask
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size)
    mask = mask * logits_mask  # 除去本身

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  # 只留了同类型的
    log_prob = logits - \
               torch.log(torch.sum(exp_logits, axis=1, keepdims=True))

    # compute mean of log-likelihood over positive
    # this may introduce NaNs due to zero division,
    # when a class only has one example in the batch
    mask_sum = torch.sum(mask, axis=1)
    mean_log_prob_pos = torch.sum(
        mask * log_prob, axis=1)[mask_sum > 0] / mask_sum[mask_sum > 0]

    # loss
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    # loss = tf.reduce_mean(tf.reshape(loss, [anchor_count, batch_size]))
    loss = torch.mean(loss)
    return loss