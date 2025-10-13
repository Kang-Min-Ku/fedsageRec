import torch
import numpy as np

def linear(x, step, first_annealing_round, max_round=0, min=0.0, max=1.0):
    """
    Normally, round starts from 0.
    if max round is specified, step becomes the largest beta.
    else step becomes the increasing ratio of beta
    """
    assert first_annealing_round > 0
    if max_round == 0:
        max_round = first_annealing_round + 1
    assert max_round > first_annealing_round

    beta = (x-first_annealing_round)/(max_round-first_annealing_round) * step

    if beta <= min:
        beta = min
    if beta >= max:
        beta = max

    return beta

def sigmoid(x, temperature, first_annealing_round, max_round=0, min=0.0, max=1.0):
    """
    Normally, round starts from 0.
    """
    assert first_annealing_round > 0
    if max_round == 0:
        max_round = first_annealing_round + 1
    assert max_round > first_annealing_round

    beta = 1 / (1 + np.exp(-(x-first_annealing_round) / (max_round-first_annealing_round) / temperature)) - 0.5

    if beta <= min:
        beta = min
    if beta >= max:
        beta = max

    return beta

def exponential_increasing(x, temperature, first_annealing_round, max_round=0, min=0.0, max=1.0):
    """
    Normally, round starts from 0.
    """
    assert first_annealing_round > 0
    if max_round == 0:
        max_round = first_annealing_round + 1
    assert max_round > first_annealing_round

    beta = 1 - np.exp(-(x-first_annealing_round) / (max_round-first_annealing_round) / temperature)
    
    if beta <= min:
        beta = min
    if beta >= max:
        beta = max

    return beta

def exponential_decreasing(x, temperature, first_annealing_round, max_round=0, min=0.0, max=1.0):
    """
    Normally, round starts from 0.
    """
    assert first_annealing_round > 0
    if max_round == 0:
        max_round = first_annealing_round + 1
    assert max_round > first_annealing_round

    beta = np.exp(-(x-first_annealing_round) / (max_round-first_annealing_round) / temperature)
    
    if beta <= min:
        beta = min
    if beta >= max:
        beta = max

    return beta