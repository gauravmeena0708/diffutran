import torch
import torch.nn as nn
import torch.nn.functional as F

class OTCFM(nn.Module):
    """
    Mixed Continuous-Discrete Flow Matching.
    Numericals: Continuous Limit (OT-CFM).
    Categoricals: Discrete Classifier-Free Guidance DDPM.
    """
    def __init__(self, model, tokenizer, config):
        """
        Args:
            model: DualStreamTransformer
            tokenizer: QuantileTokenizer (with Decoders)
            config: DSFlowConfig
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.sigma_min = getattr(config, 'sigma_min', 1e-4)
        self.reconstruction_loss_weight = getattr(config, 'reconstruction_loss_weight', 0.1)

    def compute_loss(self, x_num, x_cat, y=None):
        """
        Compute joint Flow Matching + Cross-Entropy Loss.
        Args:
            x_num: [B, N_num] or [B, T, N_num] Float
            x_cat: [B, N_cat] or [B, T, N_cat] Long
            y: [B] Long (Optional Class Labels)
        Returns:
            loss: scalar tensor
            logs: dict of metrics
        """
        device = x_num.device if x_num.nelement() > 0 else x_cat.device
        is_temporal = (x_num.dim() == 3) or (x_cat.dim() == 3)
        
        if is_temporal:
            B, T, _ = x_num.shape if x_num.dim() == 3 else x_cat.shape
            batch_size = B
        else:
            batch_size = x_num.shape[0] if x_num.nelement() > 0 else x_cat.shape[0]
            
        t = torch.rand(batch_size, device=device)
        # For temporal, we might want different t per step? 
        # But for simplicity, we use same t for the whole sequence per batch.
        t_expanded = t.view(-1, 1, 1) # [B, 1, 1]
        if is_temporal:
             t_expanded = t_expanded.unsqueeze(1) # [B, 1, 1, 1]

        N_num = len(self.tokenizer.numerical_cols) if hasattr(self.tokenizer, 'numerical_cols') else 0
        N_cat = len(self.tokenizer.categorical_cols) if hasattr(self.tokenizer, 'categorical_cols') else 0

        # 1. Forward transform to get the target clean tokens x1
        x1_all = self.tokenizer.forward(x_num, x_cat) # [B, N, D] or [B, T, N, D]
        x1_num = x1_all[..., :N_num, :] if N_num > 0 else None
        
        xt_list = []
        ut_num = None

        # 2. Continuous Flow for Numericals
        if x1_num is not None:
            x0_num = torch.randn_like(x1_num)
            xt_num = t_expanded * x1_num + (1.0 - t_expanded) * x0_num
            ut_num = x1_num - x0_num
            xt_list.append(xt_num)
            
        # 3. Discrete Flow for Categoricals
        if N_cat > 0:
            xt_cat_idx = x_cat.clone()
            # For categoricals, we need to iterate columns. 
            # If temporal, x_cat is [B, T, N_cat]
            for i, col in enumerate(self.tokenizer.categorical_cols):
                cardinality = self.tokenizer.cardinalities[col]
                # Probability of keeping original token is t
                # We need to match shape of x_cat[:, i] or x_cat[:, :, i]
                t_slice = t if not is_temporal else t.view(-1, 1)
                keep_mask = torch.rand(xt_cat_idx[..., i].shape, device=device) < t_slice
                random_class = torch.randint(0, cardinality, xt_cat_idx[..., i].shape, device=device)
                xt_cat_idx[..., i] = torch.where(keep_mask, xt_cat_idx[..., i], random_class)
            
            # Embed perturbed categorical tokens
            # We must flatten temporal if present for cat_embeddings call
            if is_temporal:
                xt_cat_idx_flat = xt_cat_idx.view(B * T, N_cat)
            else:
                xt_cat_idx_flat = xt_cat_idx
                
            xt_cat_emb = []
            for i, col in enumerate(self.tokenizer.categorical_cols):
                emb = self.tokenizer.cat_embeddings[
                    self.tokenizer._module_key(col)
                ](xt_cat_idx_flat[:, i]).unsqueeze(1)
                xt_cat_emb.append(emb)
            xt_cat_emb = torch.cat(xt_cat_emb, dim=1)
            
            if is_temporal:
                xt_cat_emb = xt_cat_emb.view(B, T, N_cat, -1)
                
            xt_list.append(xt_cat_emb)

        xt = torch.cat(xt_list, dim=-2) if len(xt_list) > 1 else xt_list[0]
        
        if y is not None:
             y = y.to(device)
             
        # 4. Neural Network Predicts Velocity & Logits
        vt = self.model(xt, t, y)
        
        # 5. Compute Losses
        loss_flow_num = torch.tensor(0.0, device=device)
        loss_ce_cat = torch.tensor(0.0, device=device)
        
        if N_num > 0:
            pred_v_num = vt[..., :N_num, :]
            loss_flow_num = F.mse_loss(pred_v_num, ut_num)
            
        if N_cat > 0:
            pred_cat_emb = vt[..., N_num:, :]
            # Flatten for CE loss
            if is_temporal:
                pred_cat_emb = pred_cat_emb.reshape(B * T, N_cat, -1)
                x_cat_target = x_cat.reshape(B * T, N_cat)
            else:
                x_cat_target = x_cat
                
            for i, col in enumerate(self.tokenizer.categorical_cols):
                logits = self.tokenizer.cat_decoders[
                    self.tokenizer._module_key(col)
                ](pred_cat_emb[:, i, :])
                target = x_cat_target[:, i].to(device)
                loss_ce_cat += F.cross_entropy(logits, target)
            loss_ce_cat /= N_cat
        
        # 6. Reconstruction Loss
        loss_rec = torch.tensor(0.0, device=device)
        if self.reconstruction_loss_weight > 0:
            decoded = self.tokenizer.decode(x1_all)
            
            # Reconstruction loss for numericals
            if N_num > 0:
                for i, col in enumerate(self.tokenizer.numerical_cols):
                    pred_val = decoded[col]  
                    true_val = x_num[..., i:i+1]
                    loss_rec += F.mse_loss(pred_val, true_val)
                loss_rec /= N_num
            
            # Reconstruction loss for categoricals
            if N_cat > 0:
                for i, col in enumerate(self.tokenizer.categorical_cols):
                    logits = decoded[col]  
                    target = x_cat[..., i]
                    # Reshape for CE if temporal
                    if is_temporal:
                        logits = logits.reshape(-1, logits.shape[-1])
                        target = target.reshape(-1)
                    loss_rec += F.cross_entropy(logits, target)
                loss_rec /= N_cat
            
        loss = loss_flow_num + loss_ce_cat + self.reconstruction_loss_weight * loss_rec
        
        return loss, {
            'loss': loss.item(),
            'loss_flow_num': loss_flow_num.item() if N_num > 0 else 0.0,
            'loss_ce_cat': loss_ce_cat.item() if N_cat > 0 else 0.0,
            'loss_rec': loss_rec.item() if self.reconstruction_loss_weight > 0 else 0.0
        }

    @torch.no_grad()
    def sample(self, n_samples, steps=20, y=None, w_max=3.0,
               cat_constraints=None, num_constraints=None, solver='dpm++'):
        """
        Generate samples using DPM-Solver++ (Continuous) & Discrete DDPM, with Adaptive CFG.

        Parameters
        ----------
        solver : str
            'euler' or 'dpm++' (2nd-order Midpoint ODE solver).
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        N_num = len(self.tokenizer.numerical_cols) if hasattr(self.tokenizer, 'numerical_cols') else 0
        N_cat = len(self.tokenizer.categorical_cols) if hasattr(self.tokenizer, 'categorical_cols') else 0
        d_model = self.config.d_model
        
        # Initialize
        xt_num = torch.randn(n_samples, N_num, d_model, device=device) if N_num > 0 else None
        xt_cat_idx = torch.zeros(n_samples, N_cat, dtype=torch.long, device=device)
        
        if N_cat > 0:
            for i, col in enumerate(self.tokenizer.categorical_cols):
                cardinality = self.tokenizer.cardinalities[col]
                xt_cat_idx[:, i] = torch.randint(0, cardinality, (n_samples,), device=device)
                
        if y is not None:
            y = y.to(device)
            if len(y) != n_samples:
                 raise ValueError("Length of y must match n_samples")

        # Pre-encode categorical constraint values to class indices
        pinned_cat = {}  # {col_idx: LongTensor [n_samples]}
        if cat_constraints:
            for col, val in cat_constraints.items():
                if col not in self.tokenizer.categorical_cols:
                    continue
                j = self.tokenizer.categorical_cols.index(col)
                if self.tokenizer.is_hashed.get(col, False):
                    class_idx = hash(str(val)) % self.tokenizer.max_cardinality
                else:
                    le = self.tokenizer.label_encoders[col]
                    class_idx = int(le.transform([str(val)])[0])
                pinned_cat[j] = torch.full((n_samples,), class_idx,
                                           dtype=torch.long, device=device)

        # Pre-compute numerical constraint embeddings once
        pinned_num = {}  # {col_idx: FloatTensor [d_model]}
        if num_constraints:
            for col, val in num_constraints.items():
                if col not in self.tokenizer.numerical_cols:
                    continue
                i_col = self.tokenizer.numerical_cols.index(col)
                qt = self.tokenizer.quantile_transformers[col]
                val_qt = float(qt.transform([[float(val)]])[0, 0])
                val_t  = torch.tensor([[val_qt]], device=device, dtype=torch.float32)
                target_emb = self.tokenizer.num_projections[
                    self.tokenizer._module_key(col)
                ](val_t).squeeze(0)  # [d_model]
                pinned_num[i_col] = target_emb

        # Apply pins to initial state
        for j, idx_tensor in pinned_cat.items():
            xt_cat_idx[:, j] = idx_tensor
        for i_col, emb in pinned_num.items():
            if xt_num is not None:
                xt_num[:, i_col, :] = emb

        times = torch.linspace(0, 1, steps + 1, device=device)
        
        def get_velocity(xt_n, xt_c_idx, t_val, y_val):
            t_batch = torch.full((n_samples,), t_val, device=device)
            xt_list = []
            if N_num > 0:
                xt_list.append(xt_n)
            if N_cat > 0:
                xt_cat_emb = []
                for j, col in enumerate(self.tokenizer.categorical_cols):
                    emb = self.tokenizer.cat_embeddings[
                        self.tokenizer._module_key(col)
                    ](xt_c_idx[:, j]).unsqueeze(1)
                    xt_cat_emb.append(emb)
                xt_list.append(torch.cat(xt_cat_emb, dim=1))
            
            xt_full = torch.cat(xt_list, dim=1) if len(xt_list) > 1 else xt_list[0]
            
            if y_val is not None:
                vt_cond = self.model(xt_full, t_batch, y_val)
                vt_uncond = self.model(xt_full, t_batch, None)
                w_t = w_max * (1.0 - t_val)
                return vt_uncond + w_t * (vt_cond - vt_uncond)
            else:
                return self.model(xt_full, t_batch, None)

        # Adaptive Integration Loop
        for i in range(steps):
            t_curr = times[i].item()
            t_next = times[i+1].item()
            dt = t_next - t_curr
            
            # Continuous Solver Step (Numerical)
            if N_num > 0:
                vt = get_velocity(xt_num, xt_cat_idx, t_curr, y)
                v_num = vt[:, :N_num, :]
                
                if solver == 'dpm++' and i < steps - 1:
                    # 2nd-order Midpoint Step
                    xt_mid = xt_num + v_num * (dt * 0.5)
                    # Re-pin for intermediate step
                    for i_col, emb in pinned_num.items():
                        xt_mid[:, i_col, :] = emb
                    
                    vt_mid = get_velocity(xt_mid, xt_cat_idx, t_curr + dt*0.5, y)
                    v_num_mid = vt_mid[:, :N_num, :]
                    xt_num = xt_num + v_num_mid * dt
                else:
                    # 1st-order Euler Step
                    xt_num = xt_num + v_num * dt
                
                # Re-pin constrained numerical columns
                for i_col, emb in pinned_num.items():
                    xt_num[:, i_col, :] = emb
            else:
                # Still need vt for categorical updates
                vt = get_velocity(None, xt_cat_idx, t_curr, y)

            # Discrete Reverse DDPM Step (Categorical)
            if N_cat > 0:
                pred_cat_emb = vt[:, N_num:, :]
                for j, col in enumerate(self.tokenizer.categorical_cols):
                    logits = self.tokenizer.cat_decoders[
                        self.tokenizer._module_key(col)
                    ](pred_cat_emb[:, j, :])
                    probs = torch.softmax(logits, dim=-1)
                    x1_sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    keep_mask = torch.rand(n_samples, device=device) < t_next
                    cardinality = self.tokenizer.cardinalities[col]
                    random_class = torch.randint(0, cardinality, (n_samples,), device=device)
                    xt_cat_idx[:, j] = torch.where(keep_mask, x1_sampled, random_class)

                for j, idx_tensor in pinned_cat.items():
                    xt_cat_idx[:, j] = idx_tensor
                    
        # Return final state: numericals as ODE endpoint embeddings,
        # categoricals as last velocity outputs — inverse_transform expects
        # cat_decoders(vt_cat) → logits, not cat_embeddings(idx).
        final_list = []
        if N_num > 0:
            final_list.append(xt_num)
        if N_cat > 0:
            final_list.append(vt[:, N_num:, :])

        return torch.cat(final_list, dim=1) if len(final_list) > 1 else final_list[0]

if __name__ == "__main__":
    from dataclasses import dataclass
    @dataclass
    class DummyConfig:
        d_model: int = 64
        sigma_min: float = 1e-4

    class DummyTokenizer(nn.Module):
        def __init__(self):
            super().__init__()
            self.numerical_cols = ['n1', 'n2']
            self.categorical_cols = ['c1']
            self.cardinalities = {'c1': 5}
            self.cat_embeddings = nn.ModuleDict({'c1': nn.Embedding(5, 64)})
            self.cat_decoders = nn.ModuleDict({'c1': nn.Linear(64, 5)})
            self.num_projections = nn.ModuleDict({'n1': nn.Linear(1, 64), 'n2': nn.Linear(1, 64)})
        def forward(self, x_num, x_cat):
            return torch.randn(x_num.shape[0], 3, 64)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(64, 64)
        def forward(self, x, t, y=None):
            return self.layer(x)

    t_model = DummyModel()
    t_cf = OTCFM(t_model, DummyTokenizer(), DummyConfig())
    x_n = torch.randn(2, 2)
    x_c = torch.tensor([[0], [1]])
    loss, logs = t_cf.compute_loss(x_n, x_c)
    assert loss.shape == torch.Size([]), "Loss should be a scalar"
    out_samp = t_cf.sample(n_samples=2, steps=2)
    assert out_samp.shape == (2, 3, 64), "Sample should output combined tokens"
    print("OTCFM Verification Passed!")
