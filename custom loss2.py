def customLoss(y_pred, y_true):
    # y_pred should be a list of numbers. Each representing how many shares to buy/sell each day
    # negative values for selling, positive for buying.
    capital = 100000.0
    heldShares = 0.0
    for i in range(0, K.int_shape(y_pred)[1]):
        # can only buy shares if capital > 0
        if (capital > 0 and y_pred[..., i] > 0):
            capital -= y_true[..., i] * y_pred[..., i]
            heldShares += 1
        # can only sell shares if shares are already owned
        if (y_pred[..., i] < 0 and heldShares > 0):
            capital -= y_true[..., i] * y_pred[..., i]
            heldShares -= y_pred[..., i]
    # sell all residual shares to get capital on last day
    capital += heldShares * y_true[..., -1]
    ROI = ((capital - 100000) / 100000)
    return ROI * -1
