from torchmetrics.functional import char_error_rate


def computer_cer(preds, targets) -> float:
    return char_error_rate(preds, targets).item()
