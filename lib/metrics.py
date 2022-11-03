import jax
import jax.numpy as jnp


def compute_premetrics(y_true, y_pred):
  y_pred = y_pred > 0

  return dict(
    TP = jnp.sum((y_pred == 1) & (y_true == 1)),
    TN = jnp.sum((y_pred == 0) & (y_true == 0)),
    FP = jnp.sum((y_pred == 1) & (y_true == 0)),
    FN = jnp.sum((y_pred == 0) & (y_true == 1)),
  )


def compute_metrics(running_stats):
  TP = running_stats['TP']
  TN = running_stats['TN']
  FP = running_stats['FP']
  FN = running_stats['FN']

  metrics = {
      'Accuracy': (TP + TN) / (TP + TN + FP + FN),
      'IoU': (TP) / (TP + FP + FN),
      'Recall': (TP) / (TP + FN),
      'Precision': (TP) / (TP + FP),
  }
  metrics['F1'] = 2 * (metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall'])
  return metrics
