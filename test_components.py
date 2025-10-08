"""
Test script to validate components before full training
"""

import sys
import torch
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import PIL
        import yaml
        import wandb
        from rich.console import Console
        print("✓ All core dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        return False

def test_config():
    """Test config loading"""
    print("\nTesting config loading...")
    try:
        from utils import load_config
        config = load_config("config.yml")
        
        assert 'wandb' in config
        assert 'model' in config
        assert 'data' in config
        assert 'training' in config
        
        print("✓ Config loaded successfully")
        print(f"  - Model: {config['model']['pretrained_model_name']}")
        print(f"  - Num labels: {config['model']['num_labels']}")
        print(f"  - Batch size: {config['data']['batch_size']}")
        print(f"  - Epochs: {config['training']['num_epochs']}")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_dataset():
    """Test dataset loading"""
    print("\nTesting dataset...")
    try:
        from utils import load_config
        from transformers import LayoutXLMProcessor
        from dataset_ocr import LayoutXLMDataset

        config = load_config("config.yml")
        data_config = config['data']
        model_name = config['model']['pretrained_model_name']

        train_list = data_config['train']
        if not Path(train_list).exists():
            print(f"✗ {train_list} not found")
            return False

        print("  Loading processor...")
        processor = LayoutXLMProcessor.from_pretrained(model_name, apply_ocr=False)

        dataset = LayoutXLMDataset(
            train_list,
            processor=processor,
            max_seq_length=data_config.get('max_length', 512),
            use_ocr=False,  # disable OCR for quick validation to avoid tesseract dependency
            num_labels=config['model']['num_labels']
        )

        print(f"✓ Dataset loaded: {len(dataset)} samples")

        sample = dataset[0]
        print(f"  - Sample keys: {list(sample.keys())}")
        print(f"  - input_ids shape: {sample['input_ids'].shape}")
        print(f"  - bbox shape: {sample['bbox'].shape}")
        print(f"  - attention_mask shape: {sample['attention_mask'].shape}")
        print(f"  - labels shape: {sample['labels'].shape}")
        # LayoutXLMProcessor returns 'image' instead of 'pixel_values'
        image_key = 'image' if 'image' in sample else 'pixel_values'
        print(f"  - {image_key} shape: {sample[image_key].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model():
    """Test model loading"""
    print("\nTesting model loading...")
    try:
        from transformers import LayoutXLMProcessor, AutoModelForTokenClassification
        
        print("  Loading processor...")
        processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base", apply_ocr=False)
        
        print("  Loading model...")
        model = AutoModelForTokenClassification.from_pretrained(
            "microsoft/layoutxlm-base",
            num_labels=9  # Updated to match config
        )
        
        print(f"✓ Model loaded successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Model error: {e}")
        return False

def test_forward_pass():
    """Test a dummy forward pass"""
    print("\nTesting forward pass...")
    try:
        from transformers import AutoModelForTokenClassification
        
        model = AutoModelForTokenClassification.from_pretrained(
            "microsoft/layoutxlm-base",
            num_labels=9  # Updated to match config
        )
        
        # Create dummy inputs respecting LayoutLMv2 constraints
        batch_size = 2
        seq_length = 128

        vocab_size = model.config.vocab_size
        max_position = getattr(model.config, "max_2d_position_embeddings", 1024)

        torch.manual_seed(0)

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

        x1 = torch.randint(0, max_position - 1, (batch_size, seq_length))
        y1 = torch.randint(0, max_position - 1, (batch_size, seq_length))
        width = torch.randint(1, max_position // 4, (batch_size, seq_length))
        height = torch.randint(1, max_position // 4, (batch_size, seq_length))
        x2 = torch.clamp(x1 + width, max=max_position - 1)
        y2 = torch.clamp(y1 + height, max=max_position - 1)
        bbox = torch.stack([x1, y1, x2, y2], dim=-1)

        labels = torch.randint(0, model.config.num_labels, (batch_size, seq_length))
        
        # Create dummy image tensor (batch_size, 3, 224, 224) - BGR format for LayoutLMv2
        image = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        print(f"✓ Forward pass successful")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Logits shape: {logits.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utils():
    """Test utility functions"""
    print("\nTesting utilities...")
    try:
        from utils import set_seed, AverageMeter, count_parameters
        from transformers import AutoModelForTokenClassification
        
        # Test set_seed
        set_seed(42)
        print("✓ Set seed successful")
        
        # Test AverageMeter
        meter = AverageMeter()
        meter.update(1.0)
        meter.update(2.0)
        meter.update(3.0)
        assert abs(meter.avg - 2.0) < 1e-6
        print("✓ AverageMeter working")
        
        # Test count_parameters
        model = AutoModelForTokenClassification.from_pretrained(
            "microsoft/layoutxlm-base",
            num_labels=9  # Updated to 9 classes (0-8)
        )
        params = count_parameters(model)
        assert 'total' in params
        assert 'trainable' in params
        print(f"✓ Parameter counting working: {params['total']:,} total")
        
        return True
    except Exception as e:
        print(f"✗ Utilities error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("LayoutXLM Training - Component Tests")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Dataset", test_dataset),
        ("Model", test_model),
        ("Forward Pass", test_forward_pass),
        ("Utilities", test_utils),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ Unexpected error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Ready to start training.")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
