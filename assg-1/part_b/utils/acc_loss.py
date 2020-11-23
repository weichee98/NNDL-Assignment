import numpy as np

def smooth_curve(arr, weight=0.9):
    last = arr[0]
    smoothed = list()
    for point in arr:
        val = last * weight + (1 - weight) * point
        smoothed.append(val)
        last = val
    return np.array(smoothed)

def curve_std(arr):
    n = len(arr)
    std = np.array([np.std(arr[i:]) for i in range(n)])
    return std
    
def acc_converge_epoch(arr, mean_thres=0.0005):
    curve = smooth_curve(arr)
    result = training_result(arr, mean_thres=mean_thres, mode='acc')
    ind = np.where(abs(curve - result) < mean_thres)[0]
    count = 1
    max_count = 1
    start = 0
    for i in range(1, len(ind)):
        if ind[i] - ind[i - 1] < 3:
            count += 1
            if count > max_count:
                max_count = count
                start = i - count + 1
        else:
            count = 1
    return ind[start] + 1

def loss_converge_epoch(arr, mean_thres=0.0005):
    curve = smooth_curve(arr)
    result = training_result(arr, mean_thres=mean_thres, mode='loss')
    ind = np.where(abs(curve - result) < mean_thres)[0]
    count = 1
    max_count = 1
    start = 0
    for i in range(1, len(ind)):
        if ind[i] - ind[i - 1] < 3:
            count += 1
            if count > max_count:
                max_count = count
                start = i - count + 1
        else:
            count = 1
    return ind[start] + 1
    
def training_result(arr, mode='acc', mean_thres=0.0005):
    n = len(arr)
    arr = smooth_curve(arr)
    if mode == 'loss':
        result = np.mean(arr)
        ind = np.where(arr > result)[0]
        trimmed = arr[ind[-1] + 1:]
        while np.any(np.abs(trimmed - result) > mean_thres):
            arr = trimmed
            result = np.mean(arr)
            ind = np.where(arr > result)[0]
            trimmed = arr[ind[-1] + 1:]
    else:
        result = np.mean(arr)
        ind = np.where(arr < result)[0]
        trimmed = arr[ind[-1] + 1:]
        while np.any(np.abs(trimmed - result) > mean_thres):
            arr = trimmed
            result = np.mean(arr)
            ind = np.where(arr < result)[0]
            trimmed = arr[ind[-1] + 1:]
    return result