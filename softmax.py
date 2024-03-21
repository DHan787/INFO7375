import numpy as np

def softmax(logits):

    max_logits = np.max(logits)
    exp_logits = np.exp(logits - max_logits)
    sum_exp_logits = np.sum(exp_logits)
    softmax_output = exp_logits / sum_exp_logits
    return softmax_output
