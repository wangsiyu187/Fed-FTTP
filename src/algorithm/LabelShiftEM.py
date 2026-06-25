import torch
import torch.nn.functional as F

def _is_multilabel(y):
    return (y.dim() == 2) and (y.size(1) > 1)

@torch.no_grad()
def prob_to_logit(p, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)

@torch.no_grad()
def em_labelshift_multiclass(pyx, p_y, max_iter=100, tol=1e-6, min_prob=1e-12):
    """
    pyx: [N, C] well-calibrated p(y|x) (single-label, multi-class)
    p_y: [C] source domain prior
    Returns:
      q:   [C] estimated target domain prior
      r:   [N, C] reweighted p(y|x) (normalized)
    """
    pyx = pyx.clamp(min=min_prob, max=1 - min_prob)
    p_y = p_y.clamp(min=min_prob)
    q = p_y.clone()
    for _ in range(max_iter):
        prev = q.clone()
        w = (q / p_y).unsqueeze(0)       # [1,C]
        r = pyx * w
        r = r / r.sum(dim=1, keepdim=True).clamp(min=min_prob)
        q = r.mean(dim=0)
        if torch.linalg.vector_norm(q - prev, ord=1).item() < tol:
            break
    q = q.clamp(min=min_prob, max=1 - min_prob)
    r = r.clamp(min=min_prob, max=1 - min_prob)
    return q, r

@torch.no_grad()
def em_labelshift_binary(p1_x, p1_src, max_iter=100, tol=1e-6, min_prob=1e-12):
    """
    p1_x: [N] well-calibrated p(y=1|x) (binary, for per-class EM in multi-label)
    p1_src: scalar (source-domain prior p(y=1))
    Returns:
      q1:  scalar, target-domain prior p_t(y=1)
      r1:  [N] reweighted p(y=1|x)
    """
    p1_x = p1_x.clamp(min=min_prob, max=1 - min_prob)
    p1_src = torch.clamp(p1_src, min=min_prob, max=1 - min_prob)
    q1 = p1_src.clone()
    for _ in range(max_iter):
        prev = q1.clone()
        w_pos = q1 / p1_src
        w_neg = (1 - q1) / (1 - p1_src)
        num = p1_x * w_pos
        den = num + (1 - p1_x) * w_neg
        den = den.clamp(min=min_prob)
        r1 = num / den                         # E-step
        q1 = r1.mean()                         # M-step
        if abs(q1.item() - prev.item()) < tol:
            break
    q1 = q1.clamp(min=min_prob, max=1 - min_prob)
    r1 = r1.clamp(min=min_prob, max=1 - min_prob)
    return q1, r1


# ---------- Multi-label: per-class binary calibration ----------
class BinaryVectorScaling(torch.nn.Module):
    """
    For multi-label BCE: learn per-class scale a_c and bias b_c such that sigmoid(a*logit + b) fits calibration set labels.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(num_classes))
        self.b = torch.nn.Parameter(torch.zeros(num_classes))

    def fit(self, logits, y_true, max_iter=200, tol=1e-6):
        """
        logits, y_true: [N, C]; y_true in {0,1}
        """
        assert logits.dim() == 2 and y_true.dim() == 2, "BVS requires [N,C] tensors"
        assert logits.shape == y_true.shape, "logits and y_true must have same shape"
        params = [self.a, self.b]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

        logits = logits.detach()
        y_true = y_true.detach().float()

        def closure():
            opt.zero_grad()
            z = logits * self.a + self.b      # per-class affine
            nll = F.binary_cross_entropy_with_logits(z, y_true, reduction='mean')
            nll.backward()
            return nll

        prev = float("inf")
        for _ in range(max_iter):
            loss = opt.step(closure)
            if abs(prev - loss.item()) < tol:
                break
            prev = loss.item()

    @torch.no_grad()
    def predict_proba(self, logits):
        z = logits * self.a + self.b
        return torch.sigmoid(z)


# ---------- Single/multi-label adaptive: BCTS ----------
class BCTS(torch.nn.Module):  # Bias-Corrected Temperature Scaling
    """
    Single-label (multiclass): softmax + CrossEntropy (classic BCTS)
    Multi-label: auto-fallback to BVS (per-class affine + BCE), more robust
    """
    def __init__(self, num_classes):
        super().__init__()
        # Single-label parameters
        self.T = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.zeros(num_classes))
        # Delegate to BVS for multi-label
        self._multilabel_delegate = None
        self._num_classes = num_classes
        self._is_multilabel = False  # determined during training

    def fit(self, logits, y_true, max_iter=100, tol=1e-6):
        self._is_multilabel = _is_multilabel(y_true)
        if self._is_multilabel:
            # Multi-label: fallback to BVS
            self._multilabel_delegate = BinaryVectorScaling(self._num_classes).to(logits.device)
            self._multilabel_delegate.fit(logits, y_true, max_iter=max_iter, tol=tol)
            return self

        # Single-label: BCTS
        y_true = y_true.long().view(-1)
        params = [self.T, self.bias]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

        logits = logits.detach()

        def closure():
            opt.zero_grad()
            z = logits / self.T + self.bias
            loss = F.cross_entropy(z, y_true, reduction='mean')
            loss.backward()
            return loss

        prev = float("inf")
        for _ in range(max_iter):
            loss = opt.step(closure)
            with torch.no_grad():
                self.T.data.clamp_(min=1e-3)
            if abs(prev - loss.item()) < tol:
                break
            prev = loss.item()
        return self

    @torch.no_grad()
    def predict_proba(self, logits):
        if self._is_multilabel and self._multilabel_delegate is not None:
            return self._multilabel_delegate.predict_proba(logits)
        z = logits / self.T + self.bias
        return F.softmax(z, dim=-1)


# ---------- Single/multi-label adaptive: VS ----------
class VS(torch.nn.Module):  # Vector Scaling (log-scale + bias)
    """
    Single-label (multiclass): z = logits * W + b -> softmax; CE training
    Multi-label: z = logits * W + b -> sigmoid; BCE training (per-class)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.W = torch.nn.Parameter(torch.ones(num_classes))
        self.b = torch.nn.Parameter(torch.zeros(num_classes))
        self._is_multilabel = False

    def fit(self, logits, y_true, max_iter=200, tol=1e-6):
        self._is_multilabel = _is_multilabel(y_true)
        params = [self.W, self.b]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

        logits = logits.detach()
        if self._is_multilabel:
            # Multi-label: BCE
            assert y_true.dim() == 2 and y_true.shape == logits.shape, "VS multi-label: y_true must be [N,C] with same shape as logits"
            y_true = y_true.detach().float()

            def closure():
                opt.zero_grad()
                z = logits * self.W + self.b
                loss = F.binary_cross_entropy_with_logits(z, y_true, reduction='mean')
                loss.backward()
                return loss
        else:
            # Single-label: CE
            y_true = y_true.long().view(-1)

            def closure():
                opt.zero_grad()
                z = logits * self.W + self.b
                loss = F.cross_entropy(z, y_true, reduction='mean')
                loss.backward()
                return loss

        prev = float("inf")
        for _ in range(max_iter):
            loss = opt.step(closure)
            if abs(prev - loss.item()) < tol:
                break
            prev = loss.item()
        return self

    @torch.no_grad()
    def predict_proba(self, logits):
        z = logits * self.W + self.b
        if self._is_multilabel:
            return torch.sigmoid(z)
        return F.softmax(z, dim=-1)
