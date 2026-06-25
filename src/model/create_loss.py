import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


def create_loss(name='ce',**kwargs):
    """
    loss function must be differentiable
    """
    if name == 'ce':  # cross entropy loss
        return F.cross_entropy
#========
    elif name == 'bce':  # multi-label BCE with logits
        return F.binary_cross_entropy_with_logits
#========
    elif name == 'ent':  # entropy loss, y is not used
#=========
        def _entropy(logits, _y_unused=None):  
            if logits.dim() == 2 and logits.size(1) > 1:
                p = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
                ent = -(p*torch.log(p) + (1-p)*torch.log(1-p)).mean()
                return ent
            else:
                return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1).mean()
        return _entropy
    
    elif name == 'ral':
        # Sensible defaults (following paper & ASL experience) — all overridable via args
        gamma_pos = kwargs.get('gamma_pos', 0.0)   # γ+
        gamma_neg = kwargs.get('gamma_neg', 3.0)   # γ-
        tau       = kwargs.get('tau', 0.05)        # dynamic threshold / truncation point
        lam       = kwargs.get('lam', 1.5)         # Hill slope, recommended ~1.5 by the paper
        M         = kwargs.get('M', 2)             # positive polynomial degree
        N         = kwargs.get('N', 2)             # negative polynomial degree
        # Polynomial coefficients gamma_m, gamma_n (default all 1)
        alpha     = kwargs.get('alpha', [1.0]*M)
        beta      = kwargs.get('beta',  [1.0]*N)

        alpha = torch.tensor(alpha, dtype=torch.float32)
        beta  = torch.tensor(beta,  dtype=torch.float32)

        def _ral_loss(logits: torch.Tensor, targets: torch.Tensor):
            """
            logits: [B, C], raw logits
            targets: [B, C] in {0,1} (multi-hot)
            """
            # Probability and numerical stability
            p = torch.sigmoid(logits)
            p = p.clamp(1e-6, 1 - 1e-6)          # numerical stability
            y = targets.float()

            # --- Positive loss: y * sum_m gamma_m * (1 - p)^{m + gamma+}
            one_minus_p = (1.0 - p)
            # Shape alignment: accumulate per power m
            pos = 0.0
            for m in range(1, M+1):
                pos = pos + alpha[m-1] * (one_minus_p ** (m + gamma_pos))
            pos = y * pos

            # --- Negative loss: eta(p) * (1 - y) * sum_n gamma_n * p_eta^{n + gamma-}
            p_tau = torch.clamp(p - tau, min=0.0)
            neg_poly = 0.0
            for n in range(1, N+1):
                neg_poly = neg_poly + beta[n-1] * (p_tau ** (n + gamma_neg))
            # Hill term eta(p) = tau - p
            psi = (lam - p)
            neg = (1.0 - y) * psi * neg_poly

            # Sum over classes, then average over samples
            loss = (pos + neg).sum(dim=1).mean()
            return loss

        return _ral_loss
#========
    else:
        raise NotImplementedError('Unknown loss name: %s' % name)
