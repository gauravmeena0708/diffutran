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
            x_num: [B, N_num] Float
            x_cat: [B, N_cat] Long
            y: [B] Long (Optional Class Labels)
        Returns:
            loss: scalar tensor
            logs: dict of metrics
        """
        batch_size = x_num.shape[0] if x_num.shape[1] > 0 else x_cat.shape[0]
        device = x_num.device if x_num.shape[1] > 0 else x_cat.device
        
        t = torch.rand(batch_size, device=device)
        t_expanded = t.view(-1, 1, 1)

        N_num = len(self.tokenizer.numerical_cols) if hasattr(self.tokenizer, 'numerical_cols') else 0
        N_cat = len(self.tokenizer.categorical_cols) if hasattr(self.tokenizer, 'categorical_cols') else 0

        # 1. Forward transform to get the target clean tokens x1
        x1_all = self.tokenizer.forward(x_num, x_cat) # [B, N_num + N_cat, D]
        x1_num = x1_all[:, :N_num, :] if N_num > 0 else None
        
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
            for i, col in enumerate(self.tokenizer.categorical_cols):
                cardinality = self.tokenizer.cardinalities[col]
                # Probability of keeping original token is t
                keep_mask = torch.rand(batch_size, device=device) < t
                random_class = torch.randint(0, cardinality, (batch_size,), device=device)
                xt_cat_idx[:, i] = torch.where(keep_mask, xt_cat_idx[:, i], random_class)
            
            # Embed perturbed categorical tokens
            xt_cat_emb = []
            for i, col in enumerate(self.tokenizer.categorical_cols):
                emb = self.tokenizer.cat_embeddings[
                    self.tokenizer._module_key(col)
                ](xt_cat_idx[:, i]).unsqueeze(1)
                xt_cat_emb.append(emb)
            xt_cat_emb = torch.cat(xt_cat_emb, dim=1)
            xt_list.append(xt_cat_emb)

        xt = torch.cat(xt_list, dim=1) if len(xt_list) > 1 else xt_list[0]
        
        if y is not None:
             y = y.to(device)
             
        # 4. Neural Network Predicts Velocity & Logits
        vt = self.model(xt, t, y)
        
        # 5. Compute Losses
        loss_flow_num = torch.tensor(0.0, device=device)
        loss_ce_cat = torch.tensor(0.0, device=device)
        
        if N_num > 0:
            pred_v_num = vt[:, :N_num, :]
            loss_flow_num = F.mse_loss(pred_v_num, ut_num)
            
        if N_cat > 0:
            pred_cat_emb = vt[:, N_num:, :]
            for i, col in enumerate(self.tokenizer.categorical_cols):
                logits = self.tokenizer.cat_decoders[
                    self.tokenizer._module_key(col)
                ](pred_cat_emb[:, i, :])
                target = x_cat[:, i].to(device)
                loss_ce_cat += F.cross_entropy(logits, target)
            loss_ce_cat /= N_cat
        
        # 6. Reconstruction Loss: Prevent embedding collapse by forcing decodability
        # This ensures the tokenizer embeddings remain semantically meaningful
        loss_rec = torch.tensor(0.0, device=device)
        if self.reconstruction_loss_weight > 0:
            decoded = self.tokenizer.decode(x1_all)
            
            # Reconstruction loss for numericals
            if N_num > 0:
                for i, col in enumerate(self.tokenizer.numerical_cols):
                    pred_val = decoded[col]  # [B, 1]
                    true_val = x_num[:, i:i+1]  # [B, 1]
                    loss_rec += F.mse_loss(pred_val, true_val)
                loss_rec /= N_num
            
            # Reconstruction loss for categoricals
            if N_cat > 0:
                for i, col in enumerate(self.tokenizer.categorical_cols):
                    logits = decoded[col]  # [B, cardinality]
                    target = x_cat[:, i]  # [B]
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
               cat_constraints=None, num_constraints=None):
        """
        Generate samples using DPM-Solver++ (Continuous) & Discrete DDPM, with Adaptive CFG.

        Constraint satisfaction
        -----------------------
        cat_constraints : dict {col_name: str_value} or None
            Categorical columns to hard-pin.  The string value is encoded via
            the tokenizer's LabelEncoder at the start of sampling and the
            resulting class index is forced into ``xt_cat_idx`` after every
            discrete reverse step, so the transformer always sees the correct
            token for that column.

        num_constraints : dict {col_name: float_value} or None
            Numerical columns to hard-pin (values in original data space).
            Each target scalar is quantile-transformed and projected to embedding
            space via ``num_projections[col]``, then written into ``xt_num``
            after every Euler step.
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

        # Apply pins to initial state so first model call already sees them
        for j, idx_tensor in pinned_cat.items():
            xt_cat_idx[:, j] = idx_tensor
        for i_col, emb in pinned_num.items():
            if xt_num is not None:
                xt_num[:, i_col, :] = emb

        times = torch.linspace(0, 1, steps + 1, device=device)
        
        # Adaptive Integration Loop
        for i in range(steps):
            t_curr = times[i]
            t_next = times[i+1]
            dt = t_next - t_curr
            t_batch = torch.full((n_samples,), t_curr.item(), device=device)
            
            # Prepare full xt
            xt_list = []
            if N_num > 0:
                xt_list.append(xt_num)
            if N_cat > 0:
                xt_cat_emb = []
                for j, col in enumerate(self.tokenizer.categorical_cols):
                    emb = self.tokenizer.cat_embeddings[
                        self.tokenizer._module_key(col)
                    ](xt_cat_idx[:, j]).unsqueeze(1)
                    xt_cat_emb.append(emb)
                xt_list.append(torch.cat(xt_cat_emb, dim=1))
                
            xt = torch.cat(xt_list, dim=1) if len(xt_list) > 1 else xt_list[0]
            
            # Adaptive CFG Evaluation
            if y is not None:
                vt_cond = self.model(xt, t_batch, y)
                vt_uncond = self.model(xt, t_batch, None)
                w_t = w_max * (1.0 - t_curr.item()) # Decay from w_max to 0
                vt = vt_uncond + w_t * (vt_cond - vt_uncond)
            else:
                vt = self.model(xt, t_batch, y)
                
            # DPM-Solver 1st Order Step for Continuous Numericals
            if N_num > 0:
                pred_v_num = vt[:, :N_num, :]
                xt_num = xt_num + pred_v_num * dt
                # Re-pin constrained numerical columns after Euler step
                for i_col, emb in pinned_num.items():
                    xt_num[:, i_col, :] = emb

            # Discrete Reverse DDPM Step for Categoricals
            if N_cat > 0:
                pred_cat_emb = vt[:, N_num:, :]
                for j, col in enumerate(self.tokenizer.categorical_cols):
                    logits = self.tokenizer.cat_decoders[
                        self.tokenizer._module_key(col)
                    ](pred_cat_emb[:, j, :])
                    probs = torch.softmax(logits, dim=-1) # [B, Card]

                    x1_sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

                    keep_mask = torch.rand(n_samples, device=device) < t_next.item()
                    cardinality = self.tokenizer.cardinalities[col]
                    random_class = torch.randint(0, cardinality, (n_samples,), device=device)
                    xt_cat_idx[:, j] = torch.where(keep_mask, x1_sampled, random_class)

                # Re-pin constrained categorical columns after discrete update
                for j, idx_tensor in pinned_cat.items():
                    xt_cat_idx[:, j] = idx_tensor
                    
        # Return final state which should be fully decoded to x1
        final_list = []
        if N_num > 0:
            final_list.append(xt_num)
        if N_cat > 0:
            xt_cat_emb = []
            for j, col in enumerate(self.tokenizer.categorical_cols):
                emb = self.tokenizer.cat_embeddings[
                    self.tokenizer._module_key(col)
                ](xt_cat_idx[:, j]).unsqueeze(1)
                xt_cat_emb.append(emb)
            final_list.append(torch.cat(xt_cat_emb, dim=1))
            
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
