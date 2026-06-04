"""
Training utilities: loss functions, training loops, and self-training.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


# =============================================================================
# Loss functions
# =============================================================================

class FocalLoss(nn.Module):
    """Focal loss for handling hard / rare examples."""

    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1 - pt) ** self.gamma * ce).mean()


# =============================================================================
# Single-attention-type training (for ablation / comparison)
# =============================================================================

def train_attention_model(model, model_name, train_loader, test_loader,
                          num_classes=13, device=None, epochs=10, lr=1e-4,
                          save_dir='../checkpoints'):
    """
    Train a single attention model (Temporal / Protocol / Feature / CrossModal)
    with multi-task outputs (binary + multiclass + anomaly).

    Args:
        model: PyTorch model with forward(features, protocols, temporal, output_type).
        model_name: String identifier for saving checkpoints and logging.
        train_loader: DataLoader yielding multi-task batches.
        test_loader: DataLoader for validation.
        num_classes: Number of multiclass labels (used for logging only).
        device: torch.device.
        epochs: Number of training epochs.
        lr: Learning rate.

    Returns a dict with training history and best metrics.
    """
    if device is None:
        device = next(model.parameters()).device
    criterions = {
        'binary': nn.CrossEntropyLoss(),
        'multiclass': nn.CrossEntropyLoss(),
        'anomaly': nn.MSELoss(),
    }
    task_weights = {'binary': 0.3, 'multiclass': 0.5, 'anomaly': 0.2}

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,
    )

    best_score = 0
    results = {'train_losses': [], 'val_metrics': [], 'best_epoch': 0, 'best_metrics': {}}
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'best_{model_name.lower()}.pth')

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0

        for batch in train_loader:
            features = batch['features'].to(device)
            protocols = batch['protocols'].to(device)
            temporal = batch['temporal'].to(device)
            bin_lab = batch['binary_labels'].to(device)
            mc_lab = batch['multiclass_labels'].to(device)
            anom_lab = batch['anomaly_scores'].to(device)

            optimizer.zero_grad()
            outputs, _ = model(features, protocols, temporal, output_type='all')

            loss = (
                task_weights['binary'] * criterions['binary'](outputs['binary'], bin_lab)
                + task_weights['multiclass'] * criterions['multiclass'](outputs['multiclass'], mc_lab)
                + task_weights['anomaly'] * criterions['anomaly'](outputs['anomaly'].squeeze(), anom_lab)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        results['train_losses'].append(avg_loss)

        # --- Validate ---
        metrics = _validate(model, test_loader, criterions, device)
        combined = 0.3 * metrics['binary_f1'] + 0.5 * metrics['multiclass_f1'] + 0.2 * metrics['anomaly_corr']
        metrics['combined_score'] = combined
        results['val_metrics'].append(metrics)
        scheduler.step(avg_loss)

        if combined > best_score:
            best_score = combined
            results['best_epoch'] = epoch
            results['best_metrics'] = metrics.copy()
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  "
              f"bin_f1={metrics['binary_f1']:.4f}  mc_f1={metrics['multiclass_f1']:.4f}  "
              f"anom={metrics['anomaly_corr']:.4f}  combined={combined:.4f}")

    return results


# =============================================================================
# Advanced ensemble training with self-training + focal loss + uncertainty
# =============================================================================

def train_advanced_ensemble(model, model_name, train_loader, test_loader,
                            num_classes, device=None, epochs=25, lr=8e-5,
                            save_dir='../checkpoints'):
    """
    Train the AdvancedEnsembleTransformer with:
    - Focal loss
    - Learned task weights
    - Uncertainty supervision (trained on prediction correctness)
    - Mixup augmentation with curriculum (features + temporal only, not protocol)

    Returns a dict with full training history.
    """
    if device is None:
        device = next(model.parameters()).device

    criterions = {
        'binary': FocalLoss(alpha=1, gamma=2),
        'multiclass': FocalLoss(alpha=1, gamma=2),
        'anomaly': nn.MSELoss(),
        'uncertainty': nn.MSELoss(),
    }

    # Learned task weights (binary, multiclass, anomaly, uncertainty)
    task_w = nn.Parameter(torch.tensor([0.25, 0.45, 0.25, 0.05],
                                       device=device, requires_grad=True))

    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': lr, 'weight_decay': 1e-5},
        {'params': [task_w], 'lr': 1e-3, 'weight_decay': 0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=6, T_mult=2, eta_min=5e-7,
    )

    best_score = 0
    results = {
        'train_losses': [], 'val_metrics': [],
        'best_epoch': 0, 'best_metrics': {},
        'task_weights_history': [],
    }
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'best_advanced_{model_name.lower()}_model.pth')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_samples = 0
        mixup_prob = min(0.4, 0.1 + (epoch / epochs) * 0.3)

        for batch in train_loader:
            features = batch['features'].to(device)
            protocols = batch['protocols'].to(device)
            temporal = batch['temporal'].to(device)
            bin_lab = batch['binary_labels'].to(device)
            mc_lab = batch['multiclass_labels'].to(device)
            anom_lab = batch['anomaly_scores'].to(device)

            # Optional mixup
            use_mixup = np.random.random() < mixup_prob
            if use_mixup:
                lam = np.random.beta(0.3, 0.3)
                idx = torch.randperm(features.size(0), device=device)
                features = lam * features + (1 - lam) * features[idx]
                # Protocol is categorical — don't interpolate, just pick one
                protocols = torch.where(
                    torch.rand(protocols.size(0), device=device).unsqueeze(-1) < lam,
                    protocols, protocols[idx],
                )
                temporal = lam * temporal + (1 - lam) * temporal[idx]
                bin_oh = lam * F.one_hot(bin_lab, 2).float() + (1 - lam) * F.one_hot(bin_lab[idx], 2).float()
                mc_oh = lam * F.one_hot(mc_lab, num_classes).float() + (1 - lam) * F.one_hot(mc_lab[idx], num_classes).float()
                anom_lab = lam * anom_lab + (1 - lam) * anom_lab[idx]

            optimizer.zero_grad()
            outputs, attn_info = model(features, protocols, temporal, output_type='all')
            tw = F.softmax(task_w, dim=0)

            # Losses
            if use_mixup:
                b_loss = -torch.mean(torch.sum(bin_oh * F.log_softmax(outputs['binary'], dim=1), dim=1))
                m_loss = -torch.mean(torch.sum(mc_oh * F.log_softmax(outputs['multiclass'], dim=1), dim=1))
            else:
                b_loss = criterions['binary'](outputs['binary'], bin_lab)
                m_loss = criterions['multiclass'](outputs['multiclass'], mc_lab)
            a_loss = criterions['anomaly'](outputs['anomaly'].squeeze(), anom_lab)

            # Uncertainty target: 1 when prediction matches label, 0 when wrong
            # This trains the model to output high confidence on correct predictions
            with torch.no_grad():
                mc_pred = torch.argmax(outputs['multiclass'], dim=1)
                u_target = (mc_pred == mc_lab).float().unsqueeze(1)
            u_loss = criterions['uncertainty'](outputs['uncertainty'], u_target)

            loss = tw[0] * b_loss + tw[1] * m_loss + tw[2] * a_loss + tw[3] * u_loss

            # Expert diversity regularization
            if 'gate_weights' in attn_info:
                loss = loss + 0.01 * torch.var(torch.mean(attn_info['gate_weights'], dim=0))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_([task_w], max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_samples += features.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        results['train_losses'].append(avg_loss)

        with torch.no_grad():
            results['task_weights_history'].append(F.softmax(task_w, dim=0).cpu().numpy())

        # Validate
        metrics = _validate_ensemble(model, test_loader, device)
        ub = 1.0 - np.mean(metrics['uncertainty_values'])
        combined = (0.3 * metrics['binary_f1'] + 0.5 * metrics['multiclass_f1']
                    + 0.2 * metrics['anomaly_corr']) * (1.0 + 0.05 * ub)
        metrics['combined_score'] = combined
        metrics['uncertainty_bonus'] = ub
        results['val_metrics'].append(metrics)

        if combined > best_score:
            best_score = combined
            results['best_epoch'] = epoch
            results['best_metrics'] = metrics.copy()
            torch.save(model.state_dict(), save_path)

        w = F.softmax(task_w, dim=0)
        print(f"Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  "
              f"bin_f1={metrics['binary_f1']:.4f}  mc_f1={metrics['multiclass_f1']:.4f}  "
              f"anom={metrics['anomaly_corr']:.4f}  combined={combined:.4f}")

    print(f"\nBest combined={best_score:.4f} at epoch {results['best_epoch']}")
    return results


# =============================================================================
# Validation helpers
# =============================================================================

def _validate(model, loader, criterions, device):
    model.eval()
    preds = {'binary': [], 'multiclass': [], 'anomaly': []}
    targets = {'binary': [], 'multiclass': [], 'anomaly': []}

    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            protocols = batch['protocols'].to(device)
            temporal = batch['temporal'].to(device)

            outputs, _ = model(features, protocols, temporal, output_type='all')

            preds['binary'].extend(torch.argmax(outputs['binary'], 1).cpu().numpy())
            preds['multiclass'].extend(torch.argmax(outputs['multiclass'], 1).cpu().numpy())
            preds['anomaly'].extend(outputs['anomaly'].squeeze().cpu().numpy())

            targets['binary'].extend(batch['binary_labels'].numpy())
            targets['multiclass'].extend(batch['multiclass_labels'].numpy())
            targets['anomaly'].extend(batch['anomaly_scores'].numpy())

    bf1 = f1_score(targets['binary'], preds['binary'], average='weighted')
    mf1 = f1_score(targets['multiclass'], preds['multiclass'], average='weighted')
    try:
        ac = float(np.corrcoef(preds['anomaly'], targets['anomaly'])[0, 1])
        if np.isnan(ac):
            ac = 0.0
    except Exception:
        ac = 0.0

    return {'binary_f1': bf1, 'multiclass_f1': mf1, 'anomaly_corr': ac}


def _validate_ensemble(model, loader, device):
    model.eval()
    preds = {'binary': [], 'multiclass': [], 'anomaly': []}
    targets = {'binary': [], 'multiclass': [], 'anomaly': []}
    uncertainties = []

    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            protocols = batch['protocols'].to(device)
            temporal = batch['temporal'].to(device)

            outputs, _ = model(features, protocols, temporal, output_type='all')

            preds['binary'].extend(torch.argmax(outputs['binary'], 1).cpu().numpy())
            preds['multiclass'].extend(torch.argmax(outputs['multiclass'], 1).cpu().numpy())
            preds['anomaly'].extend(outputs['anomaly'].squeeze().cpu().numpy())
            uncertainties.extend(outputs['uncertainty'].squeeze().cpu().numpy())

            targets['binary'].extend(batch['binary_labels'].numpy())
            targets['multiclass'].extend(batch['multiclass_labels'].numpy())
            targets['anomaly'].extend(batch['anomaly_scores'].numpy())

    bf1 = f1_score(targets['binary'], preds['binary'], average='weighted')
    mf1 = f1_score(targets['multiclass'], preds['multiclass'], average='weighted')
    try:
        ac = float(np.corrcoef(preds['anomaly'], targets['anomaly'])[0, 1])
        if np.isnan(ac):
            ac = 0.0
    except Exception:
        ac = 0.0

    return {
        'binary_f1': bf1, 'multiclass_f1': mf1, 'anomaly_corr': ac,
        'uncertainty_values': uncertainties,
    }


# =============================================================================
# Single-task multiclass training (for clean attention comparison in NB03)
# =============================================================================

def train_singletask_model(model, model_name, train_loader, test_loader,
                           num_classes=13, device=None, epochs=30, lr=1e-4,
                           save_dir='../checkpoints'):
    """
    Train a single attention model with multiclass output only.
    100% of the gradient is focused on attack-type classification,
    giving a clean signal for comparing attention mechanisms.
    Used in NB03; NB03b extends this to multi-task learning.
    """
    if device is None:
        device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,
    )
    best_f1 = 0
    results = {'train_losses': [], 'val_f1': [], 'best_epoch': 0, 'best_f1': 0}
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'best_singletask_{model_name.lower()}.pth')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            features  = batch['features'].to(device)
            protocols = batch['protocols'].to(device)
            temporal  = batch['temporal'].to(device)
            mc_lab    = batch['multiclass_labels'].to(device)

            optimizer.zero_grad()
            outputs, _ = model(features, protocols, temporal, output_type='multiclass')
            loss = criterion(outputs['multiclass'], mc_lab)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        results['train_losses'].append(avg_loss)
        mc_f1 = _validate_singletask(model, test_loader, device)
        results['val_f1'].append(mc_f1)
        scheduler.step(avg_loss)

        if mc_f1 > best_f1:
            best_f1 = mc_f1
            results['best_epoch'] = epoch
            results['best_f1'] = mc_f1
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  mc_f1={mc_f1:.4f}")

    return results


def _validate_singletask(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            features  = batch['features'].to(device)
            protocols = batch['protocols'].to(device)
            temporal  = batch['temporal'].to(device)
            outputs, _ = model(features, protocols, temporal, output_type='multiclass')
            preds.extend(torch.argmax(outputs['multiclass'], 1).cpu().numpy())
            targets.extend(batch['multiclass_labels'].numpy())
    return float(f1_score(targets, preds, average='weighted', zero_division=0))
