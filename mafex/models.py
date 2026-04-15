"""
Model Wrappers for MAFEX Evaluation

Real HuggingFace model paths for Turkish LLMs:
- BERTurk: dbmdz/bert-base-turkish-cased
- YTÜ-Cosmos: ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1
- Kumru: vngrs-ai/Kumru-2B
- Aya-23: CohereForAI/aya-23-8B
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    name: str
    type: str  # "encoder" or "decoder"
    max_length: int = 512
    quantize: bool = False
    trust_remote_code: bool = True


# ==================== REAL MODEL PATHS ====================
MODEL_REGISTRY = {
    # Encoder models (BERT-based)
    "berturk": {
        "name": "dbmdz/bert-base-turkish-cased",
        "type": "encoder",
        "max_length": 512,
        "quantize": False
    },
    "berturk-sentiment": {
        "name": "savasy/bert-base-turkish-sentiment-cased",
        "type": "encoder",
        "max_length": 512,
        "quantize": False
    },
    "loodos": {
        "name": "loodos/bert-base-turkish-uncased",
        "type": "encoder", 
        "max_length": 512,
        "quantize": False
    },
    
    # Decoder models (LLM-based)
    "cosmos": {
        "name": "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1",
        "type": "decoder", 
        "max_length": 2048,
        "quantize": True
    },
    "kumru": {
        "name": "vngrs-ai/Kumru-2B",
        "type": "decoder",
        "max_length": 2048,
        "quantize": False
    },
    "turkcell": {
        "name": "TURKCELL/Turkcell-LLM-7b-v1",
        "type": "decoder",
        "max_length": 2048,
        "quantize": True
    },
    "kanarya": {
        "name": "redrussianarmy/Kanarya-2B",
        "type": "decoder",
        "max_length": 2048,
        "quantize": False
    },
    "aya": {
        "name": "CohereForAI/aya-23-8B",
        "type": "decoder",
        "max_length": 2048,
        "quantize": True
    },
    "cosmos-bert": {
        "name": "ytu-ce-cosmos/turkish-base-bert-uncased",
        "type": "encoder",
        "max_length": 512,
        "quantize": False
    }
}


class ModelWrapper(ABC):
    """Abstract base class for LLM wrappers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
    
    @abstractmethod
    def load(self) -> bool:
        """Load model and tokenizer."""
        pass
    
    def get_embeddings_layer(self):
        """Get input embeddings layer."""
        return self.model.get_input_embeddings()
    
    def to(self, device: str):
        """Move to device."""
        if not self.config.quantize:
            self.model = self.model.to(device)
        self.device = device
        return self


class BERTurkWrapper(ModelWrapper):
    """
    BERTurk - Turkish BERT model for sequence classification.
    Model: dbmdz/bert-base-turkish-cased
    """
    
    def __init__(self, num_labels: int = 2):
        super().__init__(ModelConfig(
            name="dbmdz/bert-base-turkish-cased",
            type="encoder",
            max_length=512
        ))
        self.num_labels = num_labels
    
    def load(self) -> bool:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            print(f"Loading BERTurk: {self.config.name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.name,
                num_labels=self.num_labels
            )
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"BERTurk loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"Failed to load BERTurk: {e}")
            return False
    
    def predict(self, text: str) -> Dict[str, Any]:
        encoding = self.tokenizer(
            text, return_tensors="pt", 
            padding=True, truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = outputs.logits.argmax(dim=-1).item()
        
        return {
            "predicted_class": pred,
            "probabilities": probs[0].cpu().tolist()
        }


class CosmosWrapper(ModelWrapper):
    """
    YTÜ-Cosmos Turkish Llama - Turkish instruction-tuned LLM.
    Model: ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1
    """
    
    def __init__(self, quantize: bool = True):
        super().__init__(ModelConfig(
            name="ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1",
            type="decoder",
            max_length=2048,
            quantize=quantize
        ))
    
    def load(self) -> bool:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading YTU-Cosmos: {self.config.name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            load_kwargs = {"trust_remote_code": True}
            
            if self.config.quantize:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                load_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.name, **load_kwargs
            )
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if not self.config.quantize:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("YTU-Cosmos loaded")
            return True
            
        except Exception as e:
            print(f"Failed to load Cosmos: {e}")
            return False


class KumruWrapper(ModelWrapper):
    """
    Kumru - Turkish LLM by VNGRS based on Mistral architecture.
    Model: vngrs-ai/Kumru-2B
    """
    
    def __init__(self, quantize: bool = False):
        super().__init__(ModelConfig(
            name="vngrs-ai/Kumru-2B",
            type="decoder",
            max_length=2048,
            quantize=quantize
        ))
    
    def load(self) -> bool:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading Kumru: {self.config.name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            load_kwargs = {"trust_remote_code": True}
            
            if self.config.quantize:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                load_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.name, **load_kwargs
            )
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if not self.config.quantize:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print(f"Kumru loaded on {self.device}")
            return True
            
        except Exception as e:
            print(f"Failed to load Kumru: {e}")
            return False


class AyaWrapper(ModelWrapper):
    """
    Aya-23 - Cohere's multilingual instruction model.
    Model: CohereForAI/aya-23-8B
    """
    
    def __init__(self, quantize: bool = True):
        super().__init__(ModelConfig(
            name="CohereForAI/aya-23-8B",
            type="decoder",
            max_length=2048,
            quantize=quantize
        ))
    
    def load(self) -> bool:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading Aya-23: {self.config.name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            load_kwargs = {}
            
            if self.config.quantize:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                load_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.name, **load_kwargs
            )
            
            self.device = "cuda"
            self.model.eval()
            print("Aya-23 loaded")
            return True
            
        except Exception as e:
            print(f"Failed to load Aya: {e}")
            return False


# Demo model for testing without GPU
class DemoModel(nn.Module):
    """Lightweight model for CPU testing."""
    
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 768, num_labels: int = 2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (inputs_embeds * mask).sum(1) / mask.sum(1)
        else:
            pooled = inputs_embeds.mean(1)
        
        logits = self.classifier(pooled)
        return type('Output', (), {'logits': logits})()
    
    def get_input_embeddings(self):
        return self.embeddings


class DemoModelWrapper(ModelWrapper):
    """Demo wrapper for testing without real models."""
    
    def __init__(self):
        super().__init__(ModelConfig(name="demo", type="encoder"))
    
    def load(self) -> bool:
        from transformers import AutoTokenizer
        
        print("Loading demo model...")
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.model = DemoModel()
        self.device = "cpu"
        self.model.eval()
        print("Demo model ready")
        return True


def get_model(model_type: str, **kwargs) -> ModelWrapper:
    """
    Factory function to get model wrapper.
    Supports both specific wrappers and registry-based loading.
    """
    # Specific wrappers (for backward compatibility)
    specific_classes = {
        "berturk": BERTurkWrapper,
        "cosmos": CosmosWrapper,
        "kumru": KumruWrapper,
        "aya": AyaWrapper,
        "demo": DemoModelWrapper
    }
    
    model_type = model_type.lower()
    
    # Check specific wrappers first
    if model_type in specific_classes:
        wrapper = specific_classes[model_type](**kwargs)
        wrapper.load()
        return wrapper
    
    # Check registry for other models
    if model_type in MODEL_REGISTRY:
        config = MODEL_REGISTRY[model_type]
        wrapper = GenericModelWrapper(
            model_name=config["name"],
            model_type=config["type"],
            max_length=config["max_length"],
            quantize=config.get("quantize", False)
        )
        wrapper.load()
        return wrapper
    
    available = list(specific_classes.keys()) + list(MODEL_REGISTRY.keys())
    raise ValueError(f"Unknown model: {model_type}. Available: {available}")


class GenericModelWrapper(ModelWrapper):
    """
    Generic wrapper for any model in MODEL_REGISTRY.
    Works with both encoder (BERT) and decoder (LLM) models.
    """
    
    def __init__(self, model_name: str, model_type: str, max_length: int = 512, quantize: bool = False):
        super().__init__(ModelConfig(
            name=model_name,
            type=model_type,
            max_length=max_length,
            quantize=quantize
        ))
    
    def load(self) -> bool:
        try:
            from transformers import AutoTokenizer
            
            print(f"Loading {self.config.name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.name, 
                trust_remote_code=True
            )
            
            # Load model based on type
            if self.config.type == "encoder":
                return self._load_encoder()
            else:
                return self._load_decoder()
                
        except Exception as e:
            print(f"Failed to load {self.config.name}: {e}")
            return False
    
    def _load_encoder(self) -> bool:
        """Load encoder (BERT-like) model."""
        from transformers import AutoModelForSequenceClassification
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.name,
            num_labels=2,
            trust_remote_code=True
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"{self.config.name} loaded on {self.device}")
        return True
    
    def _load_decoder(self) -> bool:
        """Load decoder (LLM) model."""
        from transformers import AutoModelForCausalLM
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        load_kwargs = {"trust_remote_code": True}
        
        if self.config.quantize:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                load_kwargs["device_map"] = "auto"
            except ImportError:
                print("Warning: bitsandbytes not available, loading without quantization")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.name, **load_kwargs
        )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.config.quantize:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"{self.config.name} loaded")
        return True


def list_available_models():
    """Print available models and their HuggingFace paths."""
    print("\nAvailable Models:\n")
    for key, config in MODEL_REGISTRY.items():
        print(f"  {key:18} -> {config['name']}")
        print(f"                     Type: {config['type']}, Quantize: {config.get('quantize', False)}")
    print()


if __name__ == "__main__":
    list_available_models()

