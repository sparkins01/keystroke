"""
M3 (TypeNet-style recurrent encoder) and M4 (TypeFormer-style Transformer
encoder with Gaussian range encoding), Section 3.3. Both consume the raw
sequence representation (Section 3.1) and are trained once per fold with a
triplet loss over the *population* of training-fold participants, then used,
frozen, to embed held-out test-fold participants for enrolment/verification
(Section 3.4) -- they are never fine-tuned on test identities.

This still runs on a single CPU, so hyperparameters are chosen to be as large
as is tractable there, not to match the original papers' GPU-scale training.
Random triplet sampling plateaus almost immediately (most random negatives
already satisfy the margin, so they contribute ~zero gradient): each anchor
is therefore trained against a *semi-hard* negative selected from a pool of
`neg_pool_size` candidates (FaceNet-style semi-hard mining -- the closest
negative that is still farther from the anchor than the positive is, falling
back to the single hardest candidate only when none qualify). Plain
*hardest*-negative mining was tried first and collapsed the embedding space
within one epoch (all outputs converge to ~the same point, which trivially
satisfies the margin loss); semi-hard mining is the textbook fix.
"""
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EmbeddingConfig:
    hidden_size: int = 64
    embedding_dim: int = 64
    num_layers: int = 1
    n_gaussians: int = 24
    max_duration_seconds: float = 3.0
    epochs: int = 40
    steps_per_epoch: int = 150
    batch_triplets: int = 64
    neg_pool_size: int = 6
    lr: float = 1e-3
    margin: float = 0.3
    grad_clip: float = 5.0
    patience: int = 6
    seed: int = 0


def pad_batch(seqs):
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    batch = np.zeros((len(seqs), max_len, 2), dtype=np.float32)
    mask = np.zeros((len(seqs), max_len), dtype=np.float32)
    for i, s in enumerate(seqs):
        batch[i, :len(s)] = s
        mask[i, :len(s)] = 1.0
    return torch.from_numpy(batch), torch.from_numpy(mask)


class SequenceClipper:
    """Global 1.5x IQR fence on the raw sequence's inter-event duration
    column, fit on training-fold participants only (mirrors IQRClipper but
    for the shared, password-agnostic embedding-model input)."""

    def __init__(self):
        self.lower = None
        self.upper = None

    def fit(self, durations):
        arr = np.concatenate(durations)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        self.lower = q1 - 1.5 * iqr
        self.upper = q3 + 1.5 * iqr
        return self

    def transform(self, seq):
        out = seq.copy()
        out[:, 0] = np.clip(out[:, 0], self.lower, self.upper)
        return out


class LSTMEncoder(nn.Module):
    """TypeNet-style bidirectional LSTM encoder (M3).

    Implemented as a manual loop over nn.LSTMCell rather than nn.LSTM. AMD's
    ROCm-for-Windows MIOpen RNN kernel currently crashes inside backward()
    once a C++ toolchain is present (a bundled composable-kernel header
    redefines std::forward, colliding with the real MSVC standard library) --
    LSTMCell only uses ordinary matmul/elementwise ops, which are already
    confirmed working on this GPU (via the Transformer encoder), so this
    sidesteps the broken fused-RNN kernel entirely at some throughput cost
    versus a true fused implementation.

    Tried and reverted: splitting the cell into a batched input projection
    plus a smaller sequential hidden-to-hidden step (to cut the amount of
    work inside the unavoidably-sequential per-timestep loop, since ~30% GPU
    utilization was observed here). Measured slower per epoch than plain
    LSTMCell on this GPU/ROCm build (~34s/epoch vs ~7.5s/epoch) for reasons
    that weren't clear from a single measurement -- reverted rather than
    gambling a multi-hour run on an unproven, apparently-regressive change."""

    def __init__(self, cfg: EmbeddingConfig):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_layers
        self.cells_fwd = nn.ModuleList()
        self.cells_bwd = nn.ModuleList()
        for layer in range(cfg.num_layers):
            in_size = 2 if layer == 0 else cfg.hidden_size * 2
            self.cells_fwd.append(nn.LSTMCell(in_size, cfg.hidden_size))
            self.cells_bwd.append(nn.LSTMCell(in_size, cfg.hidden_size))
        self.proj = nn.Linear(cfg.hidden_size * 2, cfg.embedding_dim)

    def _run_direction(self, cell, x, mask, reverse):
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_size)
        c = x.new_zeros(batch_size, self.hidden_size)
        outputs = [None] * seq_len
        steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        for t in steps:
            valid = mask[:, t].unsqueeze(-1).bool()
            h_new, c_new = cell(x[:, t], (h, c))
            h = torch.where(valid, h_new, h)
            c = torch.where(valid, c_new, c)
            outputs[t] = h
        return torch.stack(outputs, dim=1)

    def forward(self, x, mask):
        layer_input = x
        for layer in range(self.num_layers):
            out_fwd = self._run_direction(self.cells_fwd[layer], layer_input, mask, reverse=False)
            out_bwd = self._run_direction(self.cells_bwd[layer], layer_input, mask, reverse=True)
            layer_input = torch.cat([out_fwd, out_bwd], dim=-1)
        mask_ex = mask.unsqueeze(-1)
        pooled = (layer_input * mask_ex).sum(1) / mask_ex.sum(1).clamp(min=1e-6)
        emb = self.proj(pooled)
        return nn.functional.normalize(emb, dim=-1)


class GaussianRangeEncoding(nn.Module):
    """Encodes each scalar inter-key duration as a soft membership vector over
    K Gaussian basis functions spread across [0, max_duration] (TypeFormer)."""

    def __init__(self, n_gaussians, max_duration, sigma=None):
        super().__init__()
        centers = torch.linspace(0.0, max_duration, n_gaussians)
        self.register_buffer('centers', centers)
        self.sigma = sigma or (max_duration / n_gaussians)

    def forward(self, durations):
        # durations: (batch, seq_len, 1) -> (batch, seq_len, n_gaussians)
        diff = durations - self.centers.view(1, 1, -1)
        return torch.exp(-(diff ** 2) / (2 * self.sigma ** 2))


class TransformerEncoder(nn.Module):
    """TypeFormer-style lightweight Transformer encoder (M4)."""

    def __init__(self, cfg: EmbeddingConfig):
        super().__init__()
        self.gaussian = GaussianRangeEncoding(cfg.n_gaussians, cfg.max_duration_seconds)
        d_model = cfg.hidden_size
        self.input_proj = nn.Linear(cfg.n_gaussians + 1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=d_model * 2,
                                            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.proj = nn.Linear(d_model, cfg.embedding_dim)

    def forward(self, x, mask):
        durations = x[:, :, :1]
        indicators = x[:, :, 1:2]
        gauss = self.gaussian(durations)
        h = self.input_proj(torch.cat([gauss, indicators], dim=-1))
        h = h + self.pos_embedding[:, :h.size(1), :]
        key_padding_mask = mask == 0
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        mask_ex = mask.unsqueeze(-1)
        pooled = (h * mask_ex).sum(1) / mask_ex.sum(1).clamp(min=1e-6)
        emb = self.proj(pooled)
        return nn.functional.normalize(emb, dim=-1)


def _build_candidates(dataset, participant_ids):
    candidates = []
    for pw_id in dataset.password_ids:
        by_part = {}
        for i in dataset.by_password[pw_id]:
            pid = dataset.samples[i].participant_id
            if pid in participant_ids:
                by_part.setdefault(pid, []).append(i)
        parts_with_2 = [p for p, idxs in by_part.items() if len(idxs) >= 2]
        if len(parts_with_2) < 2:
            continue
        candidates.append((pw_id, by_part, parts_with_2))
    return candidates


def sample_triplets(dataset, participant_ids, n_triplets, rng):
    """anchor/positive: two sessions of the same participant+password.
    negative: a different participant typing the same password (zero-effort impostor).
    Used only for the (small) validation-loss check, where mining isn't needed."""
    candidates = _build_candidates(dataset, participant_ids)
    if not candidates:
        return []
    triplets = []
    for _ in range(n_triplets):
        pw_id, by_part, parts_with_2 = candidates[rng.randrange(len(candidates))]
        anchor_pid = parts_with_2[rng.randrange(len(parts_with_2))]
        a_idx, p_idx = rng.sample(by_part[anchor_pid], 2)
        other_pids = [p for p in by_part if p != anchor_pid]
        if not other_pids:
            continue
        neg_pid = other_pids[rng.randrange(len(other_pids))]
        n_idx = by_part[neg_pid][rng.randrange(len(by_part[neg_pid]))]
        triplets.append((a_idx, p_idx, n_idx))
    return triplets


def sample_triplet_candidates(dataset, candidates, n_items, neg_pool_size, rng):
    """Like sample_triplets, but returns a *pool* of neg_pool_size negative
    candidates per anchor/positive pair, so the caller can pick the hardest
    one (batch-hard mining) instead of a single random impostor."""
    if not candidates:
        return []
    items = []
    for _ in range(n_items):
        pw_id, by_part, parts_with_2 = candidates[rng.randrange(len(candidates))]
        anchor_pid = parts_with_2[rng.randrange(len(parts_with_2))]
        a_idx, p_idx = rng.sample(by_part[anchor_pid], 2)
        other_pids = [p for p in by_part if p != anchor_pid]
        if not other_pids:
            continue
        neg_idxs = []
        for _ in range(neg_pool_size):
            neg_pid = other_pids[rng.randrange(len(other_pids))]
            neg_idxs.append(by_part[neg_pid][rng.randrange(len(by_part[neg_pid]))])
        items.append((a_idx, p_idx, neg_idxs))
    return items


def _select_semi_hard_negative(emb_a, emb_p, emb_n):
    """emb_a, emb_p: (B, D); emb_n: (B, K, D). Returns (B, D): for each anchor,
    the closest negative that is still farther from the anchor than the
    positive (semi-hard); if none of the K candidates qualify, falls back to
    the single hardest (closest) candidate."""
    batch_size, k, _ = emb_n.shape
    d_p = torch.norm(emb_a - emb_p, dim=-1)
    d_n = torch.norm(emb_a.unsqueeze(1) - emb_n, dim=-1)

    semi_hard_mask = d_n > d_p.unsqueeze(1)
    masked_d_n = d_n.masked_fill(~semi_hard_mask, float('inf'))
    semi_hard_idx = masked_d_n.argmin(dim=1)
    has_semi_hard = semi_hard_mask.any(dim=1)

    hardest_idx = d_n.argmin(dim=1)
    chosen_idx = torch.where(has_semi_hard, semi_hard_idx, hardest_idx)
    return emb_n[torch.arange(batch_size), chosen_idx]


def _embed_batch(encoder, seqs, clipper, device):
    if clipper is not None:
        seqs = [clipper.transform(s) for s in seqs]
    x, mask = pad_batch(seqs)
    return encoder(x.to(device), mask.to(device))


def train_embedding_model(architecture, dataset, train_ids, val_ids, clipper, cfg: EmbeddingConfig,
                           device='cpu', verbose=True):
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    if architecture == 'lstm':
        model = LSTMEncoder(cfg).to(device)
    elif architecture == 'transformer':
        model = TransformerEncoder(cfg).to(device)
    else:
        raise ValueError(architecture)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    triplet_loss = nn.TripletMarginLoss(margin=cfg.margin)

    train_candidates = _build_candidates(dataset, set(train_ids))
    val_candidates = _build_candidates(dataset, set(val_ids))
    val_items = sample_triplet_candidates(dataset, val_candidates, 300, cfg.neg_pool_size,
                                           random.Random(cfg.seed + 1))
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_steps_done = 0
        for step in range(cfg.steps_per_epoch):
            items = sample_triplet_candidates(dataset, train_candidates, cfg.batch_triplets,
                                               cfg.neg_pool_size, rng)
            if not items:
                continue
            a_idxs = [it[0] for it in items]
            p_idxs = [it[1] for it in items]
            neg_idxs_flat = [n for it in items for n in it[2]]
            batch_size = len(items)

            emb_a = _embed_batch(model, [dataset.raw_sequences[i] for i in a_idxs], clipper, device)
            emb_p = _embed_batch(model, [dataset.raw_sequences[i] for i in p_idxs], clipper, device)
            emb_n_flat = _embed_batch(model, [dataset.raw_sequences[i] for i in neg_idxs_flat], clipper, device)
            emb_n = emb_n_flat.view(batch_size, cfg.neg_pool_size, -1)

            chosen_neg = _select_semi_hard_negative(emb_a, emb_p, emb_n)
            loss = triplet_loss(emb_a, emb_p, chosen_neg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_steps_done += 1
        epoch_loss = epoch_loss / max(n_steps_done, 1)

        model.eval()
        with torch.no_grad():
            if val_items:
                a_idxs = [it[0] for it in val_items]
                p_idxs = [it[1] for it in val_items]
                neg_idxs_flat = [n for it in val_items for n in it[2]]
                vb = len(val_items)
                emb_a = _embed_batch(model, [dataset.raw_sequences[i] for i in a_idxs], clipper, device)
                emb_p = _embed_batch(model, [dataset.raw_sequences[i] for i in p_idxs], clipper, device)
                emb_n_flat = _embed_batch(model, [dataset.raw_sequences[i] for i in neg_idxs_flat], clipper, device)
                emb_n = emb_n_flat.view(vb, cfg.neg_pool_size, -1)
                chosen_neg = _select_semi_hard_negative(emb_a, emb_p, emb_n)
                val_loss = triplet_loss(emb_a, emb_p, chosen_neg).item()
            else:
                val_loss = epoch_loss

        if verbose:
            print(f'  [{architecture}] epoch {epoch + 1}/{cfg.epochs} '
                  f'train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}')

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


class EmbeddingMatcher:
    """Wraps a trained encoder to present the same gallery()/score() interface
    as the classical matchers (Section 3.3's centroid gallery + Euclidean
    distance verification score)."""

    def __init__(self, name, model, clipper, device='cpu'):
        self.name = name
        self.model = model
        self.clipper = clipper
        self.device = device
        self._cache = {}

    def _embed(self, dataset, idx):
        if idx in self._cache:
            return self._cache[idx]
        with torch.no_grad():
            emb = _embed_batch(self.model, [dataset.raw_sequences[idx]], self.clipper, self.device)
        vec = emb[0].cpu().numpy()
        self._cache[idx] = vec
        return vec

    def precompute_all(self, dataset, batch_size=2048):
        """Embeds every sample in one pass, in large batches, up front.
        Trial generation previously called _embed() one sample at a time;
        on this GPU each of those tiny calls pays enough kernel-launch
        overhead that it was slower overall than CPU. A handful of large
        batched forward passes amortises that overhead away -- after this,
        trial generation is pure dict lookup, on any device."""
        self.model.eval()
        n = len(dataset.samples)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                idxs = list(range(start, min(start + batch_size, n)))
                seqs = [dataset.raw_sequences[i] for i in idxs]
                emb = _embed_batch(self.model, seqs, self.clipper, self.device)
                emb_np = emb.cpu().numpy()
                for i, idx in enumerate(idxs):
                    self._cache[idx] = emb_np[i]

    def fit(self, dataset, train_participant_ids, clipper):
        pass  # model is already trained; nothing fold-specific left to fit

    def gallery(self, dataset, clipper, sample_idxs):
        vecs = [self._embed(dataset, i) for i in sample_idxs]
        centroid = np.mean(vecs, axis=0)
        return centroid / (np.linalg.norm(centroid) + 1e-9)

    def score(self, dataset, clipper, gallery, query_idx):
        qv = self._embed(dataset, query_idx)
        return -float(np.linalg.norm(gallery - qv))  # higher = more genuine

    def score_many(self, dataset, clipper, gallery, query_idxs):
        Q = np.vstack([self._embed(dataset, i) for i in query_idxs])
        return (-np.linalg.norm(Q - gallery, axis=1)).tolist()


class EnsembleEmbeddingMatcher:
    """Averages verification *scores* from N independently-trained embedding
    models (each a separate EmbeddingMatcher with its own training seed),
    to reduce the training-seed noise the seed-robustness check measured
    (58% of M3's fold-to-fold EER spread came from training-run luck, not
    genuine population differences).

    This deliberately fuses at the *score* level, not the embedding level.
    Two independently trained triplet-loss networks have no constraint tying
    their coordinate systems together -- nothing forces "dimension 7" of
    model A's embedding space to mean the same thing as "dimension 7" of
    model B's (the loss only constrains relative distances within each
    model's own space). Averaging embedding vectors element-wise across
    models is therefore not statistically meaningful and could cancel out
    real signal. Averaging the final distance *scores* each model
    independently produces for the same (gallery, query) pair is the
    standard, well-founded approach (score-level fusion), used throughout
    the multi-biometric/multi-model fusion literature.
    """

    def __init__(self, name, matchers):
        self.name = name
        self.matchers = matchers

    def fit(self, dataset, train_participant_ids, clipper):
        pass  # each member matcher is already trained

    def precompute_all(self, dataset, batch_size=2048):
        for m in self.matchers:
            m.precompute_all(dataset, batch_size=batch_size)

    def gallery(self, dataset, clipper, sample_idxs):
        return [m.gallery(dataset, clipper, sample_idxs) for m in self.matchers]

    def score(self, dataset, clipper, gallery, query_idx):
        return self.score_many(dataset, clipper, gallery, [query_idx])[0]

    def score_many(self, dataset, clipper, gallery, query_idxs):
        per_model_scores = np.array([
            m.score_many(dataset, clipper, g, query_idxs)
            for m, g in zip(self.matchers, gallery)
        ])
        return per_model_scores.mean(axis=0).tolist()
