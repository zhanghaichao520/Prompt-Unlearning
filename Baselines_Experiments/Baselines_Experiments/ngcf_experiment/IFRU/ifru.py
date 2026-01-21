import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from tqdm import tqdm
import numpy as np

class IFRUEngine:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def get_gradients(self, dataset, norm_adj_matrix, batch_size=2048):
        """
        Compute gradients of loss on the given dataset.
        For IFRU, this is used to compute \nabla L_d.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Collect all parameters requiring grad
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        total_loss = 0.0
        # Iterate over dataset to compute full gradient (accumulated)
        # Note: Auto-diff allows accumulating gradients.
        
        # However, for large datasets, we can't do one forward pass.
        # We need to accumulate gradients over batches.
        
        # Initialize zero gradients
        param_grads = [torch.zeros_like(p) for p in params]
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for users, pos_items, neg_items in loader:
            users = users.to(self.device).long()
            pos_items = pos_items.to(self.device).long()
            neg_items = neg_items.to(self.device).long()
            
            # Forward and Loss
            # Note: We need to clear graph or use checkpoints if OOM?
            # Standard accumulation:
            # loss = calculate_loss(...)
            # grads = grad(loss, params)
            # Add to param_grads
            
            # To save memory, do backward and accumulate in p.grad?
            # Yes, model.zero_grad() is called at start.
            # We can use standard backward().
            
            loss = self.model.calculate_loss(users, pos_items, neg_items, norm_adj_matrix)
            # Scale loss by 1/N?
            # The paper defines L(\theta) as 1/|D| sum l(...).
            # The standard implementation usually sums or means batch.
            # If standard implementation is mean of batch, then we sum (mean * batch_size) and divide by total N later?
            # Let's stick to sum of gradients and normalize later if needed.
            # Actually, Eq (11): L + eps L_d + eps L_s.
            # Influence is -H^-1 (\nabla L_d + \nabla L_s).
            # If gradients are summed, H should be summed. If mean, H should be mean.
            # Consistent scaling cancels out for Newton step direction, but influences magnitude.
            # I will use sum of gradients for now (representing total influence of D_r).
            
            loss = loss * len(users) # Un-average if calculating sum
            loss.backward()
            
        # Collect gradients
        final_grads = []
        for p in params:
             if p.grad is not None:
                final_grads.append(p.grad.detach().clone())
                p.grad.zero_() # Clear for next use
             else:
                final_grads.append(torch.zeros_like(p))
                
        return final_grads

    def get_spillover_gradients(self, dc_dataset, adj_full, adj_retain, batch_size=2048):
        """
        Compute \nabla L_s = \nabla (Loss(D_c|Full) - Loss(D_c|Retain))
        """
        self.model.eval()
        self.model.zero_grad()
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        loader = torch.utils.data.DataLoader(dc_dataset, batch_size=batch_size, shuffle=False)
        
        for users, pos_items, neg_items in loader:
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)
            
            # Loss with full graph
            loss_full = self.model.calculate_loss(users, pos_items, neg_items, adj_full)
            
            # Loss with retain graph
            loss_retain = self.model.calculate_loss(users, pos_items, neg_items, adj_retain)
            
            # Difference
            loss_diff = (loss_full - loss_retain) * len(users)
            
            loss_diff.backward()
            
        final_grads = []
        for p in params:
             if p.grad is not None:
                final_grads.append(p.grad.detach().clone())
                p.grad.zero_()
             else:
                final_grads.append(torch.zeros_like(p))
        return final_grads

    def compute_hvp(self, loss, params, v):
        """
        Compute Hessian-Vector Product: \nabla(\nabla L \cdot v)
        """
        # First gradient
        grads = grad(loss, params, create_graph=True, allow_unused=True)
        
        # Flatten
        grads_flat = []
        for g, p in zip(grads, params):
            if g is None:
                grads_flat.append(torch.zeros_like(p))
            else:
                grads_flat.append(g)
                
        # GW = gradients * v
        gw = 0.
        for g, vec in zip(grads_flat, v):
            gw += torch.sum(g * vec)
            
        # Second gradient
        hvp = grad(gw, params, retain_graph=True)
        
        # Replace None with zeros
        hvp_clean = []
        for h, p in zip(hvp, params):
            if h is None:
                hvp_clean.append(torch.zeros_like(p))
            else:
                hvp_clean.append(h)
        return hvp_clean

    def inverse_hvp(self, samples_loader, norm_adj_matrix, target_grads, steps=500, lr=0.001):
        """
        Solve H t = target_grads for t using Adam optimization.
        Taking advantage of the fact that H is positive definite (usually).
        Objective: Minimize 0.5 * t^T H t - t^T target_grads
        """
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Initialize t (estimation of influence) as zeros
        t = [torch.zeros_like(p, requires_grad=True) for p in params]
        
        optimizer = optim.Adam(t, lr=lr)
        
        # Target vector (flattened conceptually)
        # We need to treat t as parameters to optimize.
        
        print(f"Solving Inverse HVP with {steps} steps...")
        
        n_total = len(samples_loader.dataset)
        batch_iter = iter(samples_loader)
        
        for i in range(steps):
            # Sample a batch for stochastic estimation of H
            try:
                batch_data = next(batch_iter)
            except StopIteration:
                batch_iter = iter(samples_loader)
                batch_data = next(batch_iter)
                
            users, pos, neg = batch_data
            users = users.to(self.device).long()
            pos = pos.to(self.device).long()
            neg = neg.to(self.device).long()
            
            n_batch = len(users)
            scale_factor = n_total / n_batch
            
            # 1. Compute H*t
            # We compute Hv using the current t
            loss = self.model.calculate_loss(users, pos, neg, norm_adj_matrix)
            
            # Loss is mean over batch. loss * n_batch is sum over batch.
            loss_sum = loss * n_batch
            
            # hvp = H_batch * t
            hvp_res = self.compute_hvp(loss_sum, params, t)
            
            # 2. Compute gradient of objective
            # We want H_total * t - v_total
            # Estimate H_total * t with (N/B) * H_batch * t
            
            optimizer.zero_grad()
            
            for j in range(len(t)):
                grad_val = scale_factor * hvp_res[j] - target_grads[j]
                # Regularization/Damping? "Add identity matrix with damping term"
                # (H + \lambda I) t = v
                # Grad = H t + \lambda t - v
                # Lower damping for RecSys to avoid drowning out structural info
                damping = 0.001 
                grad_val += damping * t[j]
                
                if t[j].grad is None:
                    t[j].grad = grad_val
                else:
                    t[j].grad.copy_(grad_val)
            
            # Monitoring convergence every 100 steps
            if i % 100 == 0:
                with torch.no_grad():
                     param_norm = sum([x.norm().item() for x in t])
                     print(f"  Step {i}: Influence Vector Norm = {param_norm:.6f}")
                    
            optimizer.step()
            
        return [x.detach() for x in t]

    def run(self, train_dataset, forget_dataset, dc_dataset, 
            adj_full, adj_retain, influence_lr=1e-3, steps=100, spillover_weight=1.0, scale=1.0):
        
        # 1. Compute direct gradient \nabla L_d
        print("Computing direct gradients...")
        grad_d = self.get_gradients(forget_dataset, adj_full)
        norm_d = sum([g.norm().item() for g in grad_d])
        print(f"Norm of Direct Gradients: {norm_d:.4f}")
        
        # 2. Compute spillover gradient \nabla L_s
        print(f"Computing spillover gradients (weight={spillover_weight:.2f})...")
        grad_s = self.get_spillover_gradients(dc_dataset, adj_full, adj_retain)
        
        # Apply scaling to spillover gradients
        if spillover_weight != 1.0:
            grad_s = [g * spillover_weight for g in grad_s]

        norm_s = sum([g.norm().item() for g in grad_s])
        print(f"Norm of Spillover Gradients (Weighted): {norm_s:.4f}")
        
        # Target vector = -(grad_d + grad_s)
        # Note: If norms are wildly different, Spillover might be dominating noise.
        target_grads = [-(gd + gs) for gd, gs in zip(grad_d, grad_s)]
        
        # 3. Inverse HVP
        # Use full training data (or subset) to estimate Hessian
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2048, shuffle=True)
        
        print("Solving Inverse HVP...")
        influence = self.inverse_hvp(train_loader, adj_full, target_grads, steps=steps, lr=influence_lr)
        
        # 4. Update parameters
        # \theta' = \theta - \frac{1}{|D|} I
        # Note: Depending on scaling. If H is sum, and I is sum based, and we assume I approximates N * (theta* - theta).
        # Actually in Eq 12: theta* - theta = - 1/|D| * |D| * H_mean^-1 * grad_mean ...
        # If we use sum formulation:
        # H_sum * \Delta = - \nabla L_sum
        # \Delta = - H_sum^-1 \nabla L_sum
        # So we simply subtract influence from parameters.
        # No 1/|D| factor if we solved H_sum * I = \nabla L_sum (target_grads).
        # Our target_grads is \nabla L_d + \nabla L_s (sum).
        # Our HVP approximates H_sum * t.
        # So `influence` 't' is \Delta.
        
        print(f"Applying influence update with scale={scale}...")
        with torch.no_grad():
            for p, delta in zip(self.model.parameters(), influence):
                if p.requires_grad:
                    p.sub_(delta * scale)
                    
        return self.model
