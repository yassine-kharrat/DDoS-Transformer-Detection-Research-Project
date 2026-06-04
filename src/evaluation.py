"""
Evaluation utilities: metrics, attention visualization, and ablation helpers.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)


# =============================================================================
# Comprehensive evaluation
# =============================================================================

def evaluate_model(model, loader, device, label_encoder=None):
    """
    Full evaluation of a multi-task model.

    Returns dict with predictions, targets, metrics, and (for ensembles)
    gate weights and uncertainty values.
    """
    model.eval()
    preds = {'binary': [], 'multiclass': [], 'anomaly': []}
    targets = {'binary': [], 'multiclass': [], 'anomaly': []}
    gate_weights_all = []
    uncertainties = []

    with torch.no_grad():
        for batch in loader:
            features = batch['features'].to(device)
            protocols = batch['protocols'].to(device)
            temporal = batch['temporal'].to(device)

            outputs, attn_info = model(features, protocols, temporal, output_type='all')

            preds['binary'].extend(torch.argmax(outputs['binary'], 1).cpu().numpy())
            preds['multiclass'].extend(torch.argmax(outputs['multiclass'], 1).cpu().numpy())
            preds['anomaly'].extend(outputs['anomaly'].squeeze().cpu().numpy())

            targets['binary'].extend(batch['binary_labels'].numpy())
            targets['multiclass'].extend(batch['multiclass_labels'].numpy())
            targets['anomaly'].extend(batch['anomaly_scores'].numpy())

            if 'gate_weights' in attn_info:
                gate_weights_all.append(attn_info['gate_weights'].cpu().numpy())
            if 'uncertainty' in outputs:
                uncertainties.extend(outputs['uncertainty'].squeeze().cpu().numpy())

    # Metrics
    bf1 = f1_score(targets['binary'], preds['binary'], average='weighted')
    mf1 = f1_score(targets['multiclass'], preds['multiclass'], average='weighted')
    try:
        ac = float(np.corrcoef(preds['anomaly'], targets['anomaly'])[0, 1])
        if np.isnan(ac):
            ac = 0.0
    except Exception:
        ac = 0.0

    bacc = float(np.mean(np.array(preds['binary']) == np.array(targets['binary'])))
    macc = float(np.mean(np.array(preds['multiclass']) == np.array(targets['multiclass'])))

    result = {
        'preds': preds,
        'targets': targets,
        # Flattened convenience keys
        'binary_preds': np.array(preds['binary']),
        'binary_targets': np.array(targets['binary']),
        'multiclass_preds': np.array(preds['multiclass']),
        'multiclass_targets': np.array(targets['multiclass']),
        'anomaly_preds': np.array(preds['anomaly']),
        'anomaly_targets': np.array(targets['anomaly']),
        # Metrics
        'binary_f1': bf1,
        'multiclass_f1': mf1,
        'anomaly_corr': ac,
        'binary_acc': bacc,
        'multiclass_acc': macc,
    }

    if gate_weights_all:
        result['gate_weights'] = np.concatenate(gate_weights_all, axis=0)
    if uncertainties:
        result['uncertainties'] = np.array(uncertainties)
        ub = 1.0 - np.mean(uncertainties)
        result['combined_score'] = (0.3 * bf1 + 0.5 * mf1 + 0.2 * ac) * (1.0 + 0.05 * ub)
    else:
        result['combined_score'] = 0.3 * bf1 + 0.5 * mf1 + 0.2 * ac

    if label_encoder is not None:
        result['classification_report'] = classification_report(
            targets['multiclass'], preds['multiclass'],
            target_names=label_encoder.classes_,
        )

    return result


# =============================================================================
# Visualization helpers
# =============================================================================

def plot_confusion_matrix(targets, preds, class_names=None, title='Confusion Matrix',
                          figsize=(10, 8), save_path=None):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_gate_weights(gate_weights, targets=None, class_names=None,
                      expert_names=None, title='Expert Gating Weights',
                      save_path=None):
    """
    Visualise how often each expert dominates, optionally broken down by class.

    Args:
        gate_weights: np.ndarray of shape (N, num_experts).
        targets: optional array of class labels for per-class breakdown.
        class_names: optional list of class name strings.
        expert_names: optional list of expert name strings.
    """
    if expert_names is None:
        expert_names = ['Temporal', 'Protocol', 'Feature']
    means = gate_weights.mean(axis=0)
    stds = gate_weights.std(axis=0)
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    ncols = 3 if targets is not None and class_names is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))

    # Bar chart of average weights
    axes[0].bar(expert_names, means, yerr=stds, capsize=5, color=colors)
    axes[0].set_ylabel('Average Gate Weight')
    axes[0].set_title('Mean Expert Weights')
    axes[0].set_ylim(0, 1)

    # Distribution
    for i, name in enumerate(expert_names):
        axes[1].hist(gate_weights[:, i], bins=30, alpha=0.6, label=name)
    axes[1].set_xlabel('Gate Weight')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Weight Distributions')
    axes[1].legend()

    # Per-class breakdown
    if ncols == 3:
        targets_arr = np.asarray(targets)
        unique_classes = np.unique(targets_arr)
        per_class_means = np.array([
            gate_weights[targets_arr == c].mean(axis=0) for c in unique_classes
        ])
        labels = [class_names[c] if c < len(class_names) else str(c) for c in unique_classes]
        x = np.arange(len(labels))
        width = 0.25
        for i, ename in enumerate(expert_names):
            axes[2].bar(x + i * width, per_class_means[:, i], width,
                        label=ename, color=colors[i], alpha=0.8)
        axes[2].set_xticks(x + width)
        axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[2].set_ylabel('Mean Gate Weight')
        axes[2].set_title('Per-Class Expert Weights')
        axes[2].legend(fontsize=8)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_uncertainty_distribution(uncertainties, targets, preds=None,
                                  title='Uncertainty Distribution', save_path=None):
    """
    Show uncertainty distributions.

    If *preds* is provided, splits by correct vs incorrect predictions.
    Otherwise splits by class label (normal vs attack).
    """
    uncertainties = np.asarray(uncertainties)
    targets = np.asarray(targets)
    fig, ax = plt.subplots(figsize=(8, 5))

    if preds is not None:
        preds = np.asarray(preds)
        correct = preds == targets
        ax.hist(uncertainties[correct], bins=30, alpha=0.6,
                label='Correct', color='green')
        ax.hist(uncertainties[~correct], bins=30, alpha=0.6,
                label='Incorrect', color='red')
    else:
        ax.hist(uncertainties[targets == 0], bins=30, alpha=0.6,
                label='Normal', color='green')
        ax.hist(uncertainties[targets == 1], bins=30, alpha=0.6,
                label='Attack', color='red')

    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_training_curves(results, title='Training Curves', save_path=None):
    """Plot loss and F1 curves from a training results dict."""
    epochs = range(1, len(results['train_losses']) + 1)
    metrics = results['val_metrics']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, results['train_losses'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    # F1 scores
    axes[1].plot(epochs, [m['binary_f1'] for m in metrics], 'g-', label='Binary F1', linewidth=2)
    axes[1].plot(epochs, [m['multiclass_f1'] for m in metrics], 'r-', label='Multiclass F1', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Validation F1 Scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Combined score
    axes[2].plot(epochs, [m['combined_score'] for m in metrics], 'm-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Combined Score')
    axes[2].set_title('Combined Score')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def print_ablation_table(results):
    """
    Print a formatted ablation comparison table.

    Args:
        results: either a list of tuples
            ``(model_name, binary_f1, multiclass_f1, anomaly_corr, params)``
            or a dict mapping model_name → results dict (must contain
            best_metrics).
    """
    header = f"{'Model':<35} {'Binary F1':>10} {'MC F1':>10} {'Anom Corr':>10} {'Params':>10}"
    print(header)
    print('-' * len(header))

    if isinstance(results, list):
        for row in results:
            name, bf1, mf1, ac = row[0], row[1], row[2], row[3]
            params = row[4] if len(row) > 4 else ''
            print(f"{name:<35} {bf1:>10.4f} {mf1:>10.4f} {ac:>10.4f} {str(params):>10}")
    else:
        for name, res in results.items():
            m = res['best_metrics']
            print(f"{name:<35} {m.get('binary_f1', 0):>10.4f} {m.get('multiclass_f1', 0):>10.4f} "
                  f"{m.get('anomaly_corr', 0):>10.4f}")
