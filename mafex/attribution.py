"""
Attribution Methods for MAFEX

Implements baseline attribution methods:
- IntegratedGradients: Axiomatic gradient-based attribution
- SHAPAttributor: SHAP values for transformers
- RandomGroupingBaseline: Control baseline with random groupings

These serve as the token-level baselines (φ_tok) that MAFEX projects
to morpheme space.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from tqdm import tqdm


class BaseAttributor(ABC):
    """Abstract base class for attribution methods."""
    
    @abstractmethod
    def attribute(
        self,
        model,
        input_ids: torch.Tensor,
        target_idx: int,
        **kwargs
    ) -> np.ndarray:
        """
        Compute attribution scores for input tokens.
        
        Args:
            model: The neural network model
            input_ids: Token IDs tensor [1, seq_len]
            target_idx: Target class/token index
            
        Returns:
            attributions: Array of shape [seq_len]
        """
        pass


class IntegratedGradients(BaseAttributor):
    """
    Integrated Gradients attribution (Sundararajan et al., 2017).
    
    Satisfies axioms:
    - Sensitivity: Non-zero gradient → non-zero attribution
    - Implementation Invariance: Same function → same attribution
    - Completeness: Sum of attributions = F(x) - F(baseline)
    
    IG(x)_i = (x_i - x'_i) × ∫₀¹ (∂F/∂x_i)(x' + α(x-x')) dα
    """
    
    def __init__(
        self,
        n_steps: int = 50,
        baseline_type: str = "zero",  # "zero", "pad", or "mask"
        internal_batch_size: int = 8
    ):
        self.n_steps = n_steps
        self.baseline_type = baseline_type
        self.internal_batch_size = internal_batch_size
    
    def attribute(
        self,
        model,
        input_ids: torch.Tensor,
        target_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        embeddings_layer: Optional[Callable] = None,
        baseline_ids: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Compute Integrated Gradients attributions.
        
        Args:
            model: Model with forward pass
            input_ids: Input token IDs [1, seq_len]
            target_idx: Target output index
            attention_mask: Optional attention mask
            embeddings_layer: Function to get embeddings from IDs
            baseline_ids: Optional custom baseline token IDs
            
        Returns:
            attributions: [seq_len] attribution scores
        """
        device = input_ids.device
        seq_len = input_ids.shape[1]
        
        # Get embeddings
        if embeddings_layer is None:
            # Default: use model's embedding layer
            embeddings_layer = model.get_input_embeddings()
        
        # Get input embeddings
        input_embeds = embeddings_layer(input_ids)  # [1, seq_len, hidden]
        
        # Create baseline
        if baseline_ids is not None:
            baseline_embeds = embeddings_layer(baseline_ids)
        else:
            baseline_embeds = self._get_baseline_embeddings(
                input_embeds, embeddings_layer, input_ids, device
            )
        
        # Compute path integral
        scaled_inputs = self._generate_scaled_inputs(
            baseline_embeds, input_embeds, self.n_steps
        )
        
        # Accumulate gradients
        total_grads = torch.zeros_like(input_embeds)
        
        for batch_start in range(0, self.n_steps, self.internal_batch_size):
            batch_end = min(batch_start + self.internal_batch_size, self.n_steps)
            batch_inputs = scaled_inputs[batch_start:batch_end]
            
            # Stack batch
            batch_embeds = torch.cat([b for b in batch_inputs], dim=0)
            
            if attention_mask is not None:
                batch_mask = attention_mask.repeat(batch_end - batch_start, 1)
            else:
                batch_mask = None
            
            # Compute gradients
            batch_grads = self._compute_gradients(
                model, batch_embeds, target_idx, batch_mask
            )
            
            # Accumulate
            for i, grad in enumerate(torch.split(batch_grads, 1)):
                total_grads += grad
        
        # Average gradients (Riemann sum approximation)
        avg_grads = total_grads / self.n_steps
        
        # Compute attributions: (x - x') * avg_grads
        diff = input_embeds - baseline_embeds
        attributions = (diff * avg_grads).sum(dim=-1)  # Sum over hidden dim
        
        # Squeeze and convert to numpy
        attributions = attributions.squeeze(0).detach().cpu().numpy()
        
        return attributions
    
    def _get_baseline_embeddings(
        self,
        input_embeds: torch.Tensor,
        embeddings_layer: Callable,
        input_ids: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Get baseline embeddings based on baseline_type."""
        if self.baseline_type == "zero":
            return torch.zeros_like(input_embeds)
        elif self.baseline_type == "pad":
            # Use PAD token (usually ID 0)
            pad_ids = torch.zeros_like(input_ids)
            return embeddings_layer(pad_ids)
        elif self.baseline_type == "mask":
            # Use MASK token (model specific)
            mask_ids = torch.full_like(input_ids, 103)  # BERT [MASK]
            return embeddings_layer(mask_ids)
        else:
            return torch.zeros_like(input_embeds)
    
    def _generate_scaled_inputs(
        self,
        baseline: torch.Tensor,
        target: torch.Tensor,
        n_steps: int
    ) -> List[torch.Tensor]:
        """Generate interpolated inputs along the path."""
        alphas = torch.linspace(0, 1, n_steps)
        scaled = []
        
        for alpha in alphas:
            scaled.append(baseline + alpha * (target - baseline))
        
        return scaled
    
    def _compute_gradients(
        self,
        model,
        embeddings: torch.Tensor,
        target_idx: int,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute gradients of target w.r.t. embeddings."""
        embeddings = embeddings.requires_grad_(True)
        
        # Forward pass
        if attention_mask is not None:
            outputs = model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask
            )
        else:
            outputs = model(inputs_embeds=embeddings)
        
        # Get target logit/probability
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs[0]
        
        # Handle different output shapes
        if logits.dim() == 3:  # [batch, seq, vocab]
            target_logits = logits[:, -1, target_idx]  # Last token
        else:  # [batch, num_classes]
            target_logits = logits[:, target_idx]
        
        # Backward
        grads = torch.autograd.grad(
            target_logits.sum(),
            embeddings,
            retain_graph=False,
            create_graph=False
        )[0]
        
        return grads


class SHAPAttributor(BaseAttributor):
    """
    SHAP values for transformer models.
    Uses partition explainer for efficiency.
    """
    
    def __init__(self, max_evals: int = 500):
        self.max_evals = max_evals
        self._explainer = None
    
    def attribute(
        self,
        model,
        input_ids: torch.Tensor,
        target_idx: int,
        tokenizer=None,
        **kwargs
    ) -> np.ndarray:
        """Compute SHAP values."""
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Run: pip install shap")
        
        if tokenizer is None:
            raise ValueError("SHAP requires tokenizer")
        
        # Convert to text
        text = tokenizer.decode(input_ids.squeeze().tolist())
        
        # Create model function
        def model_fn(texts):
            encodings = tokenizer(
                texts.tolist(), 
                return_tensors='pt', 
                padding=True,
                truncation=True
            )
            with torch.no_grad():
                outputs = model(**encodings.to(model.device))
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
            return probs.cpu().numpy()
        
        # Create explainer
        explainer = shap.Explainer(model_fn, tokenizer)
        
        # Compute SHAP values
        shap_values = explainer([text], max_evals=self.max_evals)
        
        # Extract values for target class
        attributions = shap_values.values[0][:, target_idx]
        
        return attributions


class RandomGroupingBaseline(BaseAttributor):
    """
    Random grouping baseline for ablation study.
    
    Groups tokens randomly to match morpheme-level granularity,
    demonstrating that MAFEX gains come from linguistic alignment,
    not just dimensionality reduction.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def attribute(
        self,
        model,
        input_ids: torch.Tensor,
        target_idx: int,
        token_attributions: np.ndarray,
        n_groups: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate token attributions into random groups.
        
        Args:
            token_attributions: Original token-level scores
            n_groups: Number of groups (matches K morphemes)
            
        Returns:
            group_attributions: Aggregated scores
            A_random: Random alignment matrix used
        """
        T = len(token_attributions)
        K = n_groups
        
        # Create random assignment
        assignments = self._random_partition(T, K)
        
        # Build random alignment matrix
        A_random = np.zeros((K, T), dtype=np.float32)
        for t, k in enumerate(assignments):
            A_random[k, t] = 1.0
        
        # Normalize columns
        col_sums = A_random.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        A_random = A_random / col_sums
        
        # Project
        group_attributions = A_random @ token_attributions
        
        return group_attributions, A_random
    
    def _random_partition(self, T: int, K: int) -> List[int]:
        """Randomly partition T items into K groups."""
        assignments = []
        
        # Ensure each group has at least one member
        base_assignments = list(range(K))
        self.rng.shuffle(base_assignments)
        
        # Assign remaining tokens randomly
        for t in range(T):
            if t < K:
                assignments.append(base_assignments[t])
            else:
                assignments.append(self.rng.randint(0, K))
        
        return assignments


class DeepLIFTAttributor(BaseAttributor):
    """
    DeepLIFT attribution using Captum library.
    Computes contribution scores based on difference from reference.
    """
    
    def __init__(self):
        self._deeplift = None
    
    def attribute(
        self,
        model,
        input_ids: torch.Tensor,
        target_idx: int,
        **kwargs
    ) -> np.ndarray:
        """Compute DeepLIFT attributions."""
        try:
            from captum.attr import DeepLift
        except ImportError:
            raise ImportError("Captum not installed. Run: pip install captum")
        
        # Wrap model for Captum
        embeddings_layer = model.get_input_embeddings()
        input_embeds = embeddings_layer(input_ids)
        baseline_embeds = torch.zeros_like(input_embeds)
        
        # Create DeepLIFT attributor
        def forward_func(embeds):
            outputs = model(inputs_embeds=embeds)
            logits = outputs.logits
            if logits.dim() == 3:
                return logits[:, -1, :]
            return logits
        
        dl = DeepLift(forward_func)
        
        # Compute attributions
        attributions = dl.attribute(
            input_embeds,
            baselines=baseline_embeds,
            target=target_idx
        )
        
        # Sum over hidden dimension
        attributions = attributions.sum(dim=-1).squeeze(0)
        
        return attributions.detach().cpu().numpy()


def get_attributor(method: str, **kwargs) -> BaseAttributor:
    """Factory function to get an attributor by name."""
    methods = {
        "ig": IntegratedGradients,
        "integrated_gradients": IntegratedGradients,
        "shap": SHAPAttributor,
        "deeplift": DeepLIFTAttributor,
        "random": RandomGroupingBaseline,
    }
    
    method = method.lower()
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")
    
    return methods[method](**kwargs)
