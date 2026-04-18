"""LossRegularizer — handles all regularization terms, coefficient scheduling, and history tracking.

Extracted from models/utils.py for independent testability and clear module boundaries.
"""

import numpy as np
import torch

from connectome_gnn.models.utils import get_in_features_g_phi, get_in_features_update
from connectome_gnn.utils import to_numpy


class LossRegularizer:
    """
    Handles all regularization terms, coefficient scheduling, and history tracking.

    Usage:
        regularizer = LossRegularizer(train_config, model_config, activity_column=6,
                                       plot_frequency=100, n_neurons=1000, trainer_type='signal')

        for epoch in range(n_epochs):
            regularizer.set_epoch(epoch)

            for N in range(Niter):
                regularizer.reset_iteration()

                pred, in_features, msg = model(batch, data_id=data_id, return_all=True)

                regul_loss = regularizer.compute(model, x, in_features, ids, ids_batch, edges, device)
                loss = pred_loss + regul_loss
    """

    # Components tracked in history
    COMPONENTS = [
        'W_L1', 'W_L2', 'W_sign',
        'g_phi_diff', 'g_phi_norm', 'g_phi_weight', 'f_theta_weight',
        'f_theta_zero', 'f_theta_diff', 'f_theta_msg_diff', 'f_theta_msg_sign',
        'missing_activity', 'model_a', 'model_b',
        'f_theta_linearity', 'f_theta_centering',
        'f_theta_msg_linearity',
        'embedding_cluster',
    ]

    def __init__(self, train_config, model_config, activity_column: int,
                 plot_frequency: int, n_neurons: int, trainer_type: str = 'signal',
                 dataset: str = None, type_list=None, n_neuron_types: int = 0):
        """
        Args:
            train_config: TrainingConfig with coeff_* values
            model_config: GraphModelConfig with model settings
            activity_column: Column index for activity (6 for signal, 3 for flyvis)
            plot_frequency: How often to record to history
            n_neurons: Number of neurons for normalization
            trainer_type: 'signal' or 'flyvis'
        """
        self.train_config = train_config
        self.model_config = model_config
        self.activity_column = activity_column
        self.plot_frequency = plot_frequency
        self.n_neurons = n_neurons
        self.trainer_type = trainer_type
        # type_list: (n_neurons,) long tensor on device — set by move_type_list_to_device()
        self._type_ids_device = None if type_list is None else type_list.squeeze(-1).long()
        self._n_neuron_types = n_neuron_types
        self._type_count = None  # (n_types, 1) float, precomputed in move_type_list_to_device

        # Current epoch
        self.epoch = 0
        self.Niter = 0

        # Iteration counter
        self.iter_count = 0

        # Per-iteration accumulator (GPU scalar tensors, flushed to Python floats
        # in finalize_iteration). Initialized properly in reset_iteration().
        self._iter_tracker = {}
        self._device = None

        # Epoch boundary tracking (cumulative iter_count at each epoch start)
        self.epoch_boundaries = []

        # History for plotting
        self._history = {comp: [] for comp in self.COMPONENTS}
        self._history['regul_total'] = []
        self._history['iteration'] = []

        # Cache coefficients (Python floats)
        self._coeffs = {}
        self._update_coeffs()

        # GPU tensor mirrors of _coeffs — created lazily on first reset_iteration(device)
        # and updated in-place at epoch boundaries.  Using 0-dim GPU tensors prevents
        # torch.compile from constant-folding zero coefficients at epoch 0 and avoids
        # DeviceCopy (scalar CPU→GPU copy) when non-zero values appear at epoch 1.
        self._coeff_tensors: dict = {}

        # f_theta linearity loss state (unsupervised — no gt V_rest needed)
        self._mu_activity = None
        self._sigma_activity = None

    def move_type_list_to_device(self, device):
        """Move type_ids to the training device. Call once after device is known."""
        if self._type_ids_device is not None:
            self._type_ids_device = self._type_ids_device.to(device)
            # Precompute per-type neuron counts — static shape, avoids bincount at runtime
            t = self._type_ids_device
            n = self._n_neuron_types
            self._type_count = torch.zeros(n, 1, device=device).scatter_add_(
                0, t.unsqueeze(-1), torch.ones(t.shape[0], 1, device=device)
            ).clamp(min=1)  # (n_types, 1)

    def set_activity_stats(self, x_ts, device):
        """Cache per-neuron activity statistics for linearity loss.

        Should be called once after construction, before training starts.

        Args:
            x_ts: NeuronTimeSeries with voltage field.
            device: Torch device.
        """
        from connectome_gnn.metrics import compute_activity_stats
        mu, sigma = compute_activity_stats(x_ts, device)
        self._mu_activity = to_numpy(mu).astype(np.float32)
        self._sigma_activity = to_numpy(sigma).astype(np.float32)

    def _update_coeffs(self):
        """Recompute coefficients based on current epoch.

        Weight regularization (L1 and L2) uses exponential annealing:
        coeff * (1 - exp(-rate * epoch)), controlled by regul_annealing_rate.
        With rate=0.5 (default), ramps from 0 at epoch 0 to ~0.39x at epoch 1
        to ~0.92x at epoch 5. This allows the model to learn dynamics before
        regularization pressure is applied — critical for SIREN visual field training.
        """
        tc = self.train_config
        epoch = self.epoch
        rate = tc.regul_annealing_rate

        # Exponential ramp-up annealing for weight regularization
        # rate > 0: coeff ramps up over epochs (zero at epoch 0)
        # rate = 0: full coeff from epoch 0 (no annealing)
        def anneal(coeff):
            return float(coeff * (1 - np.exp(-rate * epoch))) if rate > 0 else float(coeff)

        self._coeffs['W_L1'] = anneal(tc.coeff_W_L1)
        self._coeffs['W_L2'] = anneal(tc.coeff_W_L2)
        self._coeffs['g_phi_weight_L1'] = anneal(tc.coeff_g_phi_weight_L1)
        self._coeffs['g_phi_weight_L2'] = anneal(tc.coeff_g_phi_weight_L2)
        self._coeffs['f_theta_weight_L1'] = anneal(tc.coeff_f_theta_weight_L1)
        self._coeffs['f_theta_weight_L2'] = anneal(tc.coeff_f_theta_weight_L2)

        # Non-annealed coefficients
        self._coeffs['W_sign'] = tc.coeff_W_sign
        self._coeffs['g_phi_diff'] = tc.coeff_g_phi_diff
        self._coeffs['g_phi_norm'] = tc.coeff_g_phi_norm
        self._coeffs['f_theta_zero'] = tc.coeff_f_theta_zero
        self._coeffs['f_theta_diff'] = tc.coeff_f_theta_diff
        self._coeffs['f_theta_msg_diff'] = tc.coeff_f_theta_msg_diff
        self._coeffs['f_theta_msg_sign'] = tc.coeff_f_theta_msg_sign
        self._coeffs['missing_activity'] = tc.coeff_missing_activity
        self._coeffs['model_a'] = tc.coeff_model_a
        self._coeffs['model_b'] = tc.coeff_model_b
        self._coeffs['f_theta_linearity'] = getattr(tc, 'coeff_f_theta_linearity', 0.0)
        self._coeffs['f_theta_centering'] = getattr(tc, 'coeff_f_theta_centering', 0.0)
        self._coeffs['f_theta_msg_linearity'] = getattr(tc, 'coeff_f_theta_msg_linearity', 0.0)
        self._coeffs['embedding_cluster'] = getattr(tc, 'coeff_embedding_cluster', 0.0)

    def set_epoch(self, epoch: int, plot_frequency: int = None, Niter: int = None):
        """Set current epoch and update coefficients."""
        self.epoch = epoch
        self._update_coeffs()
        # Update GPU tensor mirrors in-place so CUDA graphs see the new values
        # at the same GPU addresses (no guard failure, no DeviceCopy).
        if self._coeff_tensors:
            for k, v in self._coeffs.items():
                self._coeff_tensors[k].fill_(float(v))
        if plot_frequency is not None:
            self.plot_frequency = plot_frequency
        if Niter is not None:
            self.Niter = Niter
        if epoch > 0:
            self.epoch_boundaries.append(self.iter_count)

    def reset_iteration(self, device=None):
        """Reset per-iteration accumulator (called once per batch, NOT per N iteration).

        Args:
            device: torch device. Required on first call, cached afterwards.
        """
        if device is not None:
            self._device = device
        self._iter_tracker = {comp: torch.zeros((), device=self._device) for comp in self.COMPONENTS}
        # Flag to ensure W_L1 is only applied once per iteration (not per batch item)
        self._W_L1_applied_this_iter = False
        # Lazily create GPU tensor mirrors of _coeffs on first call when device is known.
        # These stay at fixed GPU addresses so torch.compile CUDA graphs never see a DeviceCopy.
        if not self._coeff_tensors and self._device is not None:
            self._coeff_tensors = {
                k: torch.tensor(float(v), device=self._device) for k, v in self._coeffs.items()
            }

    def should_record(self) -> bool:
        """Check if we should record to history this iteration."""
        return (self.iter_count % self.plot_frequency == 0) or (self.iter_count == 1)

    def needs_update_regul(self) -> bool:
        """Check if update regularization is needed (update_diff, update_msg_diff, or update_msg_sign)."""
        return (self._coeffs['f_theta_diff'] > 0 or
                self._coeffs['f_theta_msg_diff'] > 0 or
                self._coeffs['f_theta_msg_sign'] > 0)

    def _add(self, name: str, term):
        """Internal: accumulate a regularization term into a GPU scalar.

        No .item() call here — values stay as tensors so this method
        is safe to call inside torch.compiled functions.
        """
        if term is None:
            return
        self._iter_tracker[name] = self._iter_tracker[name] + term.detach()

    def compute(self, model, x, in_features, ids, ids_batch, edges, device,
                xnorm=1.0, index_weight=None):
        """
        Compute all regularization terms internally.

        Args:
            model: The neural network model
            x: NeuronState — only voltage is used
            in_features: Features for f_theta (from model forward pass, can be None)
            ids: Sample indices for regularization
            ids_batch: Batch indices
            edges: Edge tensor
            device: Torch device
            xnorm: Normalization value
            index_weight: Index for W_sign computation (signal only)

        Returns:
            Total regularization loss tensor
        """
        tc = self.train_config
        mc = self.model_config
        n_neurons = self.n_neurons
        total_regul = torch.zeros((), device=device)

        # Get model W (handle multi-run case not working here)
        # For low_rank_factorization, compute W from WL @ WR to allow gradient flow
        # --- W regularization ---

        # _ct: shorthand for coefficient tensors (pre-allocated GPU scalars, updated in-place
        # at epoch boundaries via set_epoch → no DeviceCopy, no torch.compile guard failures)
        _ct = self._coeff_tensors

        low_rank = getattr(model, 'low_rank_factorization', False)
        if low_rank and hasattr(model, 'WL') and hasattr(model, 'WR'):

            if not self._W_L1_applied_this_iter:
                regul_term = (model.WL.norm(1) + model.WR) * _ct['W_L1']
                total_regul = total_regul + regul_term
                self._add('W_L1', regul_term)
                self._W_L1_applied_this_iter = True
        else:

            # W_L1: Apply only once per iteration (not per batch item)
            if not self._W_L1_applied_this_iter:
                regul_term = model.W.norm(1) * _ct['W_L1']
                total_regul = total_regul + regul_term
                self._add('W_L1', regul_term)
                self._W_L1_applied_this_iter = True

                regul_term = model.W.norm(2) * _ct['W_L2']
                total_regul = total_regul + regul_term
                self._add('W_L2', regul_term)

        # --- g_phi / f_theta weight regularization ---
        if hasattr(model, 'g_phi'):
            for param in model.g_phi.parameters():
                regul_term = param.norm(1) * _ct['g_phi_weight_L1'] + param.norm(2) * _ct['g_phi_weight_L2']
                total_regul = total_regul + regul_term
                self._add('g_phi_weight', regul_term)

        if hasattr(model, 'f_theta'):
            for param in model.f_theta.parameters():
                regul_term = param.norm(1) * _ct['f_theta_weight_L1'] + param.norm(2) * _ct['f_theta_weight_L2']
                total_regul = total_regul + regul_term
                self._add('f_theta_weight', regul_term)

        # --- f_theta_zero regularization ---
        if self._coeffs['f_theta_zero'] > 0 and hasattr(model, 'f_theta'):
            in_features_phi = get_in_features_update(rr=None, model=model, device=device)
            func_phi = model.f_theta(in_features_phi[ids].float())
            regul_term = func_phi.norm(2) * _ct['f_theta_zero']
            total_regul = total_regul + regul_term
            self._add('f_theta_zero', regul_term)

        # --- g_phi diff/norm regularization ---
        if ((self._coeffs['g_phi_diff'] > 0) | (self._coeffs['g_phi_norm'] > 0)) and hasattr(model, 'g_phi'):
            in_features_edge, in_features_edge_next = get_in_features_g_phi(x, model, mc, xnorm, n_neurons, device)

            if self._coeffs['g_phi_diff'] > 0:
                if mc.g_phi_positive:
                    msg0 = model.g_phi(in_features_edge[ids].clone().detach()) ** 2
                    msg1 = model.g_phi(in_features_edge_next[ids].clone().detach()) ** 2
                else:
                    msg0 = model.g_phi(in_features_edge[ids].clone().detach())
                    msg1 = model.g_phi(in_features_edge_next[ids].clone().detach())
                regul_term = torch.relu(msg0 - msg1).norm(2) * _ct['g_phi_diff']
                total_regul = total_regul + regul_term
                self._add('g_phi_diff', regul_term)

            if self._coeffs['g_phi_norm'] > 0:
                in_features_edge_norm = in_features_edge.clone()
                in_features_edge_norm[:, 0] = 2 * xnorm
                if mc.g_phi_positive:
                    msg_norm = model.g_phi(in_features_edge_norm[ids].clone().detach()) ** 2
                else:
                    msg_norm = model.g_phi(in_features_edge_norm[ids].clone().detach())
                # Different normalization target for signal vs flyvis
                if self.trainer_type == 'signal':
                    regul_term = (msg_norm - 1).norm(2) * _ct['g_phi_norm']
                else:  # flyvis
                    regul_term = (msg_norm - 2 * xnorm).norm(2) * _ct['g_phi_norm']
                total_regul = total_regul + regul_term
                self._add('g_phi_norm', regul_term)

        # --- W_sign (Dale's Law) regularization ---
        if self._coeffs['W_sign'] > 0 and self.epoch > 0:
            W_sign_temp = getattr(tc, 'W_sign_temperature', 10.0)
            from connectome_gnn.metrics import get_model_W
            model_W = get_model_W(model)

            if self.trainer_type == 'signal' and index_weight is not None:
                # Signal version: uses index_weight
                if self.iter_count % 4 == 0:
                    W_sign = torch.tanh(5 * model_W) # noqa: F821
                    loss_contribs = []
                    for i in range(n_neurons):
                        indices = index_weight[int(i)]
                        if indices.numel() > 0:
                            values = W_sign[indices, i]
                            std = torch.std(values, unbiased=False)
                            loss_contribs.append(std)
                    if loss_contribs:
                        regul_term = torch.stack(loss_contribs).norm(2) * _ct['W_sign']
                        total_regul = total_regul + regul_term
                        self._add('W_sign', regul_term)
            else:
                # Flyvis version: uses scatter_add
                weights = model_W.squeeze() if model_W is not None else model.W.squeeze() # noqa: F821
                source_neurons = edges[0]

                n_pos = torch.zeros(n_neurons, device=device)
                n_neg = torch.zeros(n_neurons, device=device)
                n_total = torch.zeros(n_neurons, device=device)

                pos_mask = torch.sigmoid(W_sign_temp * weights)
                neg_mask = torch.sigmoid(-W_sign_temp * weights)

                n_pos.scatter_add_(0, source_neurons, pos_mask)
                n_neg.scatter_add_(0, source_neurons, neg_mask)
                n_total.scatter_add_(0, source_neurons, torch.ones_like(weights))

                violation = torch.where(n_total > 0,
                                        (n_pos / n_total) * (n_neg / n_total),
                                        torch.zeros_like(n_total))
                regul_term = violation.sum() * _ct['W_sign']
                total_regul = total_regul + regul_term
                self._add('W_sign', regul_term)

        # Note: f_theta regularizations (f_theta_msg_diff, f_theta_msg_sign)
        # are handled by compute_update_regul() which should be called after the model forward pass.
        # Call finalize_iteration() after all regularizations are computed to record to history.

        # --- f_theta linearity loss (unsupervised, requires f_theta + a) ---
        if (self._coeffs['f_theta_linearity'] > 0
                and self._mu_activity is not None
                and hasattr(model, 'f_theta')):
            tc = self.train_config
            warmup_threshold = int(getattr(tc, 'f_theta_linearity_warmup_fraction', 0.3) * self.Niter)
            if self.iter_count > warmup_threshold:
                rampup_iters = getattr(tc, 'f_theta_linearity_rampup_iters', 200)
                rampup_weight = min(1.0, (self.iter_count - warmup_threshold) / max(rampup_iters, 1))

                from connectome_gnn.metrics import compute_f_theta_linearity_loss
                lin_loss = compute_f_theta_linearity_loss(
                    model=model,
                    n_neurons=self.n_neurons,
                    mu=self._mu_activity,
                    sigma=self._sigma_activity,
                    device=device,
                )
                lin_term = lin_loss * _ct['f_theta_linearity'] * rampup_weight
                total_regul = total_regul + lin_term
                self._add('f_theta_linearity', lin_term)

        # --- f_theta centering loss (unsupervised V_rest anchor, requires f_theta + a) ---
        if (self._coeffs['f_theta_centering'] > 0
                and self._mu_activity is not None
                and hasattr(model, 'f_theta')):
            tc = self.train_config
            warmup_threshold = int(
                getattr(tc, 'f_theta_centering_warmup_fraction', 0.3) * self.Niter)
            if self.iter_count > warmup_threshold:
                rampup_iters = getattr(tc, 'f_theta_centering_rampup_iters', 200)
                rampup_weight = min(
                    1.0,
                    (self.iter_count - warmup_threshold) / max(rampup_iters, 1))

                from connectome_gnn.metrics import compute_f_theta_centering_loss
                cent_loss = compute_f_theta_centering_loss(
                    model=model,
                    n_neurons=self.n_neurons,
                    mu=self._mu_activity,
                    device=device,
                )
                cent_term = cent_loss * _ct['f_theta_centering'] * rampup_weight
                total_regul = total_regul + cent_term
                self._add('f_theta_centering', cent_term)

        # --- f_theta msg linearity loss (penalizes nonlinear msg response) ---
        if (self._coeffs['f_theta_msg_linearity'] > 0
                and self._mu_activity is not None
                and hasattr(model, 'f_theta')):
            tc = self.train_config
            warmup_threshold = int(
                getattr(tc, 'f_theta_msg_linearity_warmup_fraction', 0.3) * self.Niter)
            if self.iter_count > warmup_threshold:
                rampup_iters = getattr(tc, 'f_theta_msg_linearity_rampup_iters', 200)
                rampup_weight = min(
                    1.0,
                    (self.iter_count - warmup_threshold) / max(rampup_iters, 1))

                from connectome_gnn.LLM_code.staging.block_03.f_theta_msg_linearity_loss import (
                    f_theta_msg_linearity_loss,
                )
                msg_lin_loss = f_theta_msg_linearity_loss(
                    model=model,
                    n_neurons=self.n_neurons,
                    mu=self._mu_activity,
                    sigma=self._sigma_activity,
                    device=device,
                )
                msg_lin_term = msg_lin_loss * _ct['f_theta_msg_linearity'] * rampup_weight
                total_regul = total_regul + msg_lin_term
                self._add('f_theta_msg_linearity', msg_lin_term)

        # --- embedding cluster regularization ---
        # Pull each neuron's embedding toward the centroid of its cell type.
        # Centroid is computed on-the-fly from model.a so it drifts freely during training.
        if (self._coeffs['embedding_cluster'] > 0
                and self._type_ids_device is not None
                and hasattr(model, 'a')
                and model.a.requires_grad):
            a = model.a  # (n_neurons, emb_dim)
            t = self._type_ids_device  # (n_neurons,) long
            emb_dim = a.shape[1]
            n_types = self._n_neuron_types
            sum_emb = torch.zeros(n_types, emb_dim, device=device)
            sum_emb.scatter_add_(0, t.unsqueeze(-1).expand(-1, emb_dim), a)
            mean_emb = sum_emb / self._type_count  # (n_types, emb_dim)
            neuron_means = mean_emb[t]  # (n_neurons, emb_dim)
            regul_term = (a - neuron_means).norm(2) * _ct['embedding_cluster']
            total_regul = total_regul + regul_term
            self._add('embedding_cluster', regul_term)

        return total_regul

    def _record_to_history(self):
        """Append current iteration values to history. Calls .item() to extract scalars."""
        n = self.n_neurons
        total = sum(v.item() for v in self._iter_tracker.values())
        self._history['regul_total'].append(total / n)
        self._history['iteration'].append(self.iter_count)
        for comp in self.COMPONENTS:
            self._history[comp].append(self._iter_tracker[comp].item() / n)

    def compute_update_regul(self, model, in_features, ids_batch, device,
                              x=None, xnorm=None, ids=None):
        """
        Compute update function regularizations (f_theta_diff, f_theta_msg_diff, f_theta_msg_sign).

        This method should be called after the model forward pass when in_features is available.

        Args:
            model: The neural network model
            in_features: Features from model forward pass
            ids_batch: Batch indices
            device: Torch device
            x: Input tensor (required for update_diff with 'generic' update_type)
            xnorm: Normalization value (required for update_diff)
            ids: Sample indices (required for update_diff)

        Returns:
            Total update regularization loss tensor
        """
        mc = self.model_config
        embedding_dim = mc.embedding_dim
        n_neurons = self.n_neurons
        total_regul = torch.zeros((), device=device)
        _ct = self._coeff_tensors

        if in_features is None:
            return total_regul

        # f_theta_diff: enforce negative slope w.r.t. state v_i (column 0)
        # Leaky integrator: f_theta should decrease when v_i increases (df/dv < 0)
        if self._coeffs['f_theta_diff'] > 0 and hasattr(model, 'f_theta'):
            pred_v = model.f_theta(in_features.clone().detach())
            in_features_v_next = in_features.clone().detach()
            delta_v = 0.05 * max(float(xnorm), 1e-6) if xnorm is not None else 1e-6
            in_features_v_next[:, 0] = in_features_v_next[:, 0] + delta_v
            pred_v_next = model.f_theta(in_features_v_next)
            # penalize positive slope: relu(f(v+dv) - f(v))
            regul_term = torch.relu(pred_v_next[ids_batch] - pred_v[ids_batch]).norm(2) * _ct['f_theta_diff']
            total_regul = total_regul + regul_term
            self._add('f_theta_diff', regul_term)

        if self._coeffs['f_theta_msg_diff'] > 0:
            pred_msg = model.f_theta(in_features.clone().detach())
            in_features_msg_next = in_features.clone().detach()
            delta_msg = 0.05 * max(float(xnorm), 1e-6) if xnorm is not None else 1e-6
            in_features_msg_next[:, embedding_dim + 1] = in_features_msg_next[:, embedding_dim + 1] + delta_msg
            pred_msg_next = model.f_theta(in_features_msg_next)
            regul_term = torch.relu(pred_msg[ids_batch] - pred_msg_next[ids_batch]).norm(2) * _ct['f_theta_msg_diff']
            total_regul = total_regul + regul_term
            self._add('f_theta_msg_diff', regul_term)

        if self._coeffs['f_theta_msg_sign'] > 0:
            in_features_modified = in_features.clone().detach()
            in_features_modified[:, 0] = 0
            pred_msg = model.f_theta(in_features_modified)
            msg_col = in_features[:, embedding_dim + 1].clone().detach()
            regul_term = (torch.tanh(pred_msg / 0.1) - torch.tanh(msg_col.unsqueeze(-1) / 0.1)).norm(2) * _ct['f_theta_msg_sign']
            total_regul = total_regul + regul_term
            self._add('f_theta_msg_sign', regul_term)

        return total_regul

    def finalize_iteration(self):
        """
        Finalize the current iteration by recording to history if appropriate.

        This should be called once per training iteration N (after all batch regularizations).
        iter_count increments here — NOT in reset_iteration() — so it counts iterations, not batches.
        """
        self.iter_count += 1
        if self.should_record():
            self._record_to_history()

    def get_iteration_total(self) -> float:
        """Get total regularization for current iteration."""
        return sum(v.item() for v in self._iter_tracker.values())

    def get_iteration_total_tensor(self) -> torch.Tensor:
        """Return total regularization as a GPU tensor (no .item() sync)."""
        return torch.stack(list(self._iter_tracker.values())).sum()

    def get_history(self) -> dict:
        """Get history dictionary for plotting."""
        return self._history
