import torch
import torch.nn as nn
import torch.nn.functional as F

class OTCFM(nn.Module):
    """
    Optimal Transport Conditional Flow Matching (OT-CFM).
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
        self.sigma_min = config.sigma_min

    def compute_loss(self, x_num, x_cat, y=None):
        """
        Compute joint Flow Matching + Reconstruction Loss.
        Args:
            x_num: [B, N_num] Float
            x_cat: [B, N_cat] Long
            y: [B] Long (Optional Class Labels)
        Returns:
            loss: scalar tensor
            logs: dict of metrics
        """
        # 1. Get Target Data Tokens (x1)
        x1 = self.tokenizer.forward(x_num, x_cat) # [B, N, D]
        batch_size, n_tokens, dim = x1.shape
        device = x1.device
        
        # 2. Sample Noise (x0)
        x0 = torch.randn_like(x1)
        
        # 3. Sample Time (t)
        t = torch.rand(batch_size, device=device)
        
        # 4. Interpolate (xt)
        # Conditional Flow Matching: path is straight line
        # mu_t = t * x1 + (1 - t) * x0
        # shape broadcasting for t: [B, 1, 1]
        t_expanded = t.view(-1, 1, 1)
        xt = t_expanded * x1 + (1.0 - t_expanded) * x0
        
        # 5. Target Velocity (ut)
        # u_t = x1 - x0 (derivative of mu_t wrt t)
        ut = x1 - x0
        
        # 6. Predict Velocity (vt)
        # Pass y (condition) if present
        if y is not None:
             y = y.to(device)
             
        vt = self.model(xt, t, y)
        
        # 7. Flow Loss (MSE)
        # Weighted? OT-CFM is uniform weight.
        loss_flow = F.mse_loss(vt, ut)
        
        # 8. Reconstruction Loss (Supervise Embeddings)
        # We try to reconstruct x_num and x_cat from x1 (the target tokens)
        # This ensures x1 actually represents the data
        decoded = self.tokenizer.decode(x1)
        
        loss_rec_num = 0.0
        loss_rec_cat = 0.0
        
        # Numerical Reconstruction (MSE in component space)
        if self.tokenizer.numerical_cols:
            for i, col in enumerate(self.tokenizer.numerical_cols):
                pred = decoded[col] # [B, 1]
                target = x_num[:, i].unsqueeze(1).to(device) # [B, 1]
                loss_rec_num += F.mse_loss(pred, target)
            loss_rec_num /= len(self.tokenizer.numerical_cols)
            
        # Categorical Reconstruction (Cross Entropy)
        if self.tokenizer.categorical_cols:
            for i, col in enumerate(self.tokenizer.categorical_cols):
                logits = decoded[col] # [B, Card]
                target = x_cat[:, i].to(device) # [B]
                loss_rec_cat += F.cross_entropy(logits, target)
            loss_rec_cat /= len(self.tokenizer.categorical_cols)
            
        loss_rec = loss_rec_num + loss_rec_cat
        
        # Total Loss
        # We might need to weigh reconstruction higher initially?
        # For now, 1.0
        loss = loss_flow + loss_rec
        
        return loss, {
            'loss': loss.item(),
            'loss_flow': loss_flow.item(),
            'loss_rec': loss_rec.item()
        }

    @torch.no_grad()
    def sample(self, n_samples, steps=20, y=None):
        """
        Generate samples using Euler ODE Solver.
        Args:
            n_samples: Number of samples to generate
            steps: Number of ODE steps
            y: [n_samples] Optional condition labels
        Returns:
            x_tokens: [B, N, D]
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        # Get shape from tokenizer
        n_cols = len(self.tokenizer.numerical_cols) + len(self.tokenizer.categorical_cols)
        d_model = self.config.d_model
        
        # 1. Sample x0 ~ N(0, 1)
        shape = (n_samples, n_cols, d_model)
        xt = torch.randn(shape, device=device)
        
        # handle y
        if y is not None:
            y = y.to(device)
            if len(y) != n_samples:
                 raise ValueError("Length of y must match n_samples")
        
        # 2. ODE Solver (0 -> 1)
        dt = 1.0 / steps
        
        # We loop from t=0 to t=1
        times = torch.linspace(0, 1, steps + 1, device=device)
        
        for i in range(steps):
            t_curr = times[i]
            # t needs to be batch vector
            t_batch = torch.full((n_samples,), t_curr.item(), device=device)
            
            # Predict velocity
            vt = self.model(xt, t_batch, y)
            
            # Euler Step
            xt = xt + vt * dt
            
        # xt is now x1 (Generated Tokens)
        return xt

