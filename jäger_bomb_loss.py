from ultralytics.utils.loss import v8DetectionLoss

class JägerBombLoss(v8DetectionLoss):
    
    def __call__(self, preds, batch):
        loss, loss_item = super().__call__(preds, batch)
        # lav ting og sager her
        return loss, loss_item