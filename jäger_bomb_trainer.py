from configs import TrainingConfig, SaveConfig
import torch
import numpy as np

class JägerBombTrainer:
    def __init__(self, cfg: TrainingConfig, save_cfg : SaveConfig):
        self.cfg = cfg
        self.save_cfg = save_cfg
        # Initialize AMP (Automatic Mixed Precision) scaler for better training
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda', enabled=True)
            self.amp = True
        else:
            self.scaler = torch.amp.GradScaler('cpu', enabled=False)
            self.amp = False
        # EMA disabled for testing
        self.ema = None
        print("⚠️ EMA (Exponential Moving Average) disabled")
        
    @torch.no_grad()
    def _validate(self, epoch):
        # Use regular model for validation (EMA disabled)
        model_to_validate = self.cfg.model
        model_to_validate.eval() 
        val_box = 0
        val_cls = 0
        val_dfl = 0
        count = 0
        for X_val, y_val in self.cfg.val_dataloader:
            X_val, y_val = X_val.to(self.cfg.device), y_val.to(self.cfg.device)

            batch = self._prepare_batch_dict(X_val, y_val)

            pred_val = model_to_validate(X_val)

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
        
        # Warmup settings (like Ultralytics)
        warmup_epochs = 3.0
        nb = len(self.cfg.train_dataloader)
        nw = max(round(warmup_epochs * nb), 100)  # number of warmup iterations
        
        for epoch in range(self.cfg.epochs):
            self.cfg.model.train(True)
            
            for batch_idx, (X, y) in enumerate(self.cfg.train_dataloader):
                # Warmup learning rate for first few epochs
                ni = batch_idx + nb * epoch  # number integrated batches
                if ni <= nw:
                    xi = [0, nw]  # warmup iteration range
                    # Warmup: gradually increase LR from 0.1 to target
                    for j, x in enumerate(self.cfg.optimizer.param_groups):
                        x['lr'] = np.interp(ni, xi, [0.1 * x['initial_lr'], x['initial_lr']])
                X, y = X.to(self.cfg.device), y.to(self.cfg.device) 
                batch = self._prepare_batch_dict(X, y)
                # Forward pass with AMP
                with torch.amp.autocast(device_type='cuda' if self.amp else 'cpu', enabled=self.amp):
                    pred = self.cfg.model.forward(X)
                    batch_loss, last_loss = self.cfg.loss_fn(pred, batch)
                
                box_loss, cls_loss, dfl_loss = last_loss.detach().cpu().numpy().round(3)
                loss = batch_loss.sum()
                
                # Backward pass with gradient scaling
                self.cfg.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping (prevents exploding gradients)
                self.scaler.unscale_(self.cfg.optimizer)
                torch.nn.utils.clip_grad_norm_(self.cfg.model.parameters(), max_norm=10.0)
                
                # Optimizer step with scaler
                self.scaler.step(self.cfg.optimizer)
                self.scaler.update()

                if batch_idx % self.cfg.log_interval == 0:
                    print(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"Box: {box_loss:.4f}, Cls: {cls_loss:.4f}, DFL: {dfl_loss:.4f}, "
                        f"Total: {(box_loss+cls_loss+dfl_loss):.4f}"
                    )

            curr_val_loss = self._validate(epoch)
            
            # Step the learning rate scheduler (only after warmup)
            if epoch >= warmup_epochs:
                self.cfg.scheduler.step()
            
            current_lr = self.cfg.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
            
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