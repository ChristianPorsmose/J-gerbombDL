from configs import TrainingConfig, SaveConfig
import torch
import numpy as np

class JägerBombTrainer:
    def __init__(self, cfg: TrainingConfig, save_cfg : SaveConfig):
        self.cfg = cfg
        self.save_cfg = save_cfg
        
    @torch.no_grad()
    def _validate(self, epoch):
        self.cfg.model.eval() 
        val_box = 0
        val_cls = 0
        val_dfl = 0
        count = 0
        for X_val, y_val in self.cfg.val_dataloader:
            X_val, y_val = X_val.to(self.cfg.device), y_val.to(self.cfg.device)

            batch = self._prepare_batch_dict(X_val, y_val)

            pred_val = self.cfg.model(X_val)

            _, last_loss = self.cfg.loss_fn(pred_val, batch)
            box_loss, cls_loss, dfl_loss = last_loss.cpu().numpy().round(3)

            val_box += box_loss
            val_cls += cls_loss
            val_dfl += dfl_loss
            count += 1

        print(
            f"VALIDATION — Epoch {epoch}: "
            f"Box: {(val_box/count):.4f}, "
            f"Cls: {(val_cls/count):.4f}, "
            f"DFL: {(val_dfl/count):.4f}, "
            f"Total: {((val_box+val_cls+val_dfl)/count):.4f}"
        )
        return ((val_box+val_cls+val_dfl)/count)

    def _prepare_batch_dict(self, images: torch.Tensor, targets: torch.Tensor) -> dict:
        valid_mask = targets[:, :, 0] != -1

        samples_with_valid_targets = valid_mask.any(dim=1)

        if not samples_with_valid_targets.any():
            return {"img": images, "batch_idx": torch.empty(0), "cls": torch.empty(0), "bboxes": torch.empty(0)}

        valid_targets = targets[valid_mask]

        target_lengths = [int(v.sum().item()) for v in valid_mask]
        batch_idx = torch.cat([torch.full((length,), i) for i, length in enumerate(target_lengths) if length > 0])

        cls = valid_targets[:, 0]
        bboxes = valid_targets[:, 1:]

        return {"img": images, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

    def _save_model(self, epoch):
        save_path = f"{self.save_cfg.save_path}.pt"
        self.cfg.yolo_model.save(save_path)
        print(f"Model saved at epoch {epoch} → {save_path}")  

    def train(self):
        self.cfg.model.to(self.cfg.device)
        best_loss = np.inf
        count = 0
        early_stoppage_count = 15
        for epoch in range(self.cfg.epochs):
            self.cfg.model.train(True)
            
            for batch_idx, (X, y) in enumerate(self.cfg.train_dataloader):
                X, y = X.to(self.cfg.device), y.to(self.cfg.device) 
                batch = self._prepare_batch_dict(X, y)
                pred = self.cfg.model.forward(X)
                batch_loss, last_loss = self.cfg.loss_fn(pred, batch)
                box_loss, cls_loss, dfl_loss = last_loss.cpu().numpy().round(3)
                # if not np.allclose([box_loss, dfl_loss], [0.0, 0.0]):
                #     self.cfg.optimizer.zero_grad()
                #     batch_loss.sum().backward()
                #     self.cfg.optimizer.step()
                
                self.cfg.optimizer.zero_grad()
                # sanity check: loss must require grad
                if not batch_loss.requires_grad:
                    raise RuntimeError("batch_loss does not require grad — check loss_fn implementation (should return a tensor connected to model parameters).")
                if not np.allclose([box_loss, dfl_loss], [0.0, 0.0]):
                    batch_loss.sum().backward()
                    self.cfg.optimizer.step()

                if batch_idx % self.cfg.log_interval == 0:
                    print(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"Box: {box_loss:.4f}, Cls: {cls_loss:.4f}, DFL: {dfl_loss:.4f}, "
                        f"Total: {(box_loss+cls_loss+dfl_loss):.4f}"
                    )

            curr_val_loss = self._validate(epoch)
            if(curr_val_loss<best_loss):
                self._save_model(epoch)
                best_loss = curr_val_loss
                count = 0
            else:
                count+=1
            if(count >= early_stoppage_count):
                break
            #if (epoch + 1) % self.save_cfg.save_epoch_interval == 0:
            #    self._save_model(epoch)