# Copyright 2021 (author: Meng Wu)

import torch

def MVG(inputs, left_context=1, right_context=1):
    """
        Do moving average.
    """
    length = inputs.shape[0]
    result = []
    for idx in range(left_context, length-right_context):
        temp = torch.mean(inputs[idx-left_context:idx+right_context+1], dim=0)
        result.append(temp)
    
    result = torch.stack(result)

    return result


def test_mvg():
    x = torch.arange(100, dtype=torch.float)
    x = x.reshape(25, 4)
    print(x)
    print(MVG(x))


if __name__ == "__main__":
    test_mvg()
