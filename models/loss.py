import torch
from torch.nn import functional as F
from models.utils import bos_mask, eos_mask

def cross_entropy(x_, y_, *big_endian_length_weight):
    b_, t_, c_ = x_.shape
    losses = F.cross_entropy(x_.view(-1, c_), y_.view(-1), reduction = 'none')
    if big_endian_length_weight:
        big_endian, length, weight = big_endian_length_weight
        if big_endian:
            m_ = eos_mask(t_, length)
        else:
            m_ = bos_mask(t_, length)
        if weight is not None:
            m_ = m_ * weight
        losses = losses * m_.view(-1)
    return losses.sum() # TODO turn off non-train gradient tracking

def binary_cross_entropy(x, y, w):
    losses = F.binary_cross_entropy(x, y.type(x.dtype), reduction = 'none')
    if w is not None:
        losses = losses * w
    return losses.sum()

def hinge_loss(x, y, w):
    ones = torch.ones_like(x)
    y = torch.where(y, ones, -ones)
    losses = 1 - (x * y)
    hinge = losses < 0
    if w is not None:
        if w.is_floating_point():
            losses = losses * w
        else:
            hinge |= ~ w
    losses[hinge] = 0
    return losses.sum()

def get_decision(argmax, logits):
    if argmax:
        return logits.argmax(dim = 2)
    return logits.argmin(dim = 2)

def get_decision_with_value(score_fn, logits):
    prob, arg = sorted_decisions_with_values(score_fn, 1, logits)
    arg .squeeze_(dim = 2)
    prob.squeeze_(dim = 2)
    return prob, arg

def sorted_decisions(argmax, topk, logits):
    return logits.topk(topk, largest = argmax)[1]

def sorted_decisions_with_values(score_fn, topk, logits):
    return score_fn(logits).topk(topk)

def get_loss(net, argmax, logits, batch, *big_endian_height_mask_key):
    if argmax:
        if len(big_endian_height_mask_key) == 1:
            key, = big_endian_height_mask_key
            return cross_entropy(logits, batch[key]) # endian does not matter
        big_endian, height_mask, weight, key = big_endian_height_mask_key
        return cross_entropy(logits, batch[key], big_endian, height_mask, weight)

    if len(big_endian_height_mask_key) == 1:
        key, = big_endian_height_mask_key
        distance = net.distance(logits, batch[key])
    else:
        big_endian, height_mask, key = big_endian_height_mask_key
        if big_endian:
            height_mask = eos_mask(logits.shape[1], height_mask)
        else:
            height_mask = bos_mask(logits.shape[1], height_mask)
        distance = net.distance(logits, batch[key]) # [b, s]
        distance = distance * height_mask

    return distance.sum() + net.repulsion()