"""
Training script for LayoutXLM Document Layout Analysis
Fine-tunes LayoutXLM model for object detection on document layouts
"""

import os
import argparse
import math
import json
from typing import List
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from transformers import (
    LayoutXLMProcessor,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)
from rich.panel import Panel
import wandb
from tqdm import tqdm

from dataset_ocr import LayoutXLMDataset, collate_fn_layoutxlm
from utils import (
    load_config,
    set_seed,
    setup_logging,
    count_parameters,
    save_checkpoint,
    AverageMeter
)

console = Console()


class LayoutXLMTrainer:
    """Trainer class for LayoutXLM fine-tuning"""

    def __init__(self, config_path: str = "config.yml"):
        """Initialize trainer with configuration and core components"""
        # Load configuration
        self.config = load_config(config_path)

        # Setup logging
        self.logger = setup_logging(
            self.config['logging']['log_dir'],
            self.config['logging'].get('log_level', 'INFO')
        )

        # Set random seed
        set_seed(self.config['training']['seed'])

        # Setup device and mixed precision utilities
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        self.use_amp = self.config['training'].get('fp16', False) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Initialize state trackers
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.training_start_time = None
        self.training_end_time = None
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }

        # Initialize components
        self._init_wandb()
        self._init_model()
        self._init_data()
        self._init_optimizer()

        console.print(Panel.fit(
            "[bold green]✓ Trainer Initialized Successfully[/bold green]",
            title="LayoutXLM Training"
        ))

    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        wandb_config = self.config.get('wandb', {})

        api_key = wandb_config.get('api_key')
        if api_key:
            wandb.login(key=api_key)

        init_params = {
            'project': wandb_config.get('project', 'layoutxlm-training'),
            'name': wandb_config.get('name', 'layoutxlm-run'),
            'job_type': wandb_config.get('job_type', 'train'),
            'config': self.config
        }

        if wandb_config.get('entity'):
            init_params['entity'] = wandb_config['entity']

        wandb.init(**init_params)
        self.logger.info("Wandb initialized")

    def _init_model(self):
        """Load processor and model"""
        model_config = self.config['model']

        console.print("[yellow]Loading LayoutXLM model...[/yellow]")

        self.processor = LayoutXLMProcessor.from_pretrained(
            model_config['pretrained_model_name'],
            apply_ocr=False  # We provide words and boxes manually
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_config['pretrained_model_name'],
            num_labels=model_config['num_labels'],
            id2label=model_config['id2label']
        )

        self.model.to(self.device)

        # Prepare label mappings for metrics and logging
        id2label_config = model_config.get('id2label') or {}
        if id2label_config:
            self.id2label = {int(k): v for k, v in id2label_config.items()}
        else:
            self.id2label = {
                int(label_id): label_name
                for label_id, label_name in self.model.config.id2label.items()
            }

        self.label2id = {label: idx for idx, label in self.id2label.items()}
        self.model.config.id2label = self.id2label
        self.model.config.label2id = self.label2id

        param_counts = count_parameters(self.model)

        table = Table(title="Model Information", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Model", model_config['pretrained_model_name'])
        table.add_row("Num Labels", str(model_config['num_labels']))
        table.add_row("Total Parameters", f"{param_counts['total']:,}")
        table.add_row("Trainable Parameters", f"{param_counts['trainable']:,}")
        table.add_row("Frozen Parameters", f"{param_counts['frozen']:,}")

        console.print(table)
        self.logger.info(f"Model loaded: {model_config['pretrained_model_name']}")
        self.logger.info(f"Total parameters: {param_counts['total']:,}")

    def train_epoch(self, epoch: int):
        """Run one training epoch"""
        train_config = self.config['training']
        logging_config = self.config.get('logging', {})

        grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
        max_grad_norm = train_config.get('max_grad_norm', 1.0)
        log_steps = max(1, logging_config.get('log_steps', 100))

        self.model.train()

        loss_meter = AverageMeter()
        correct_tokens = 0
        total_tokens = 0

        use_amp = getattr(self, 'use_amp', False)
        scaler = getattr(self, 'scaler', None)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:

            task = progress.add_task(
                f"[cyan]Epoch {epoch}/{train_config['num_epochs']}",
                total=len(self.train_loader)
            )

            self.optimizer.zero_grad()

            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    bbox = batch['bbox'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    image = batch['pixel_values'].to(self.device)  # LayoutXLM uses 'image' parameter
                    labels = batch['labels'].to(self.device)
                    
                    # Clamp labels to valid range [0, num_labels-1], ignore padding (-100)
                    valid_label_mask = labels != -100
                    labels = torch.where(
                        valid_label_mask,
                        torch.clamp(labels, 0, self.model.config.num_labels - 1),
                        labels
                    )

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        outputs = self.model(
                            input_ids=input_ids,
                            bbox=bbox,
                            image=image,  # LayoutXLM uses 'image' not 'pixel_values'
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss

                    loss_value = loss.item()
                    logits = outputs.logits

                    predictions = torch.argmax(logits, dim=-1)
                    valid_mask = labels != -100
                    valid_labels = labels[valid_mask]
                    valid_predictions = predictions[valid_mask]
                    valid_total = valid_labels.numel()

                    if valid_total > 0:
                        correct_tokens += (valid_predictions == valid_labels).sum().item()
                        total_tokens += valid_total

                    scaled_loss = loss / grad_accum_steps

                    if use_amp and scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    should_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == len(self.train_loader))

                    if should_step:
                        if use_amp and scaler is not None:
                            scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                        if use_amp and scaler is not None:
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            self.optimizer.step()

                        self.scheduler.step()
                        self.optimizer.zero_grad()

                    loss_meter.update(loss_value, max(valid_total, 1))

                    if (batch_idx + 1) % log_steps == 0:
                        lr = self.optimizer.param_groups[0]['lr']
                        running_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

                        wandb.log({
                            'train/loss': loss_meter.avg,
                            'train/accuracy': running_accuracy,
                            'train/learning_rate': lr,
                            'train/epoch': epoch,
                            'train/step': self.global_step
                        })

                        self.logger.info(
                            f"Epoch [{epoch}/{train_config['num_epochs']}] "
                            f"Step [{batch_idx + 1}/{len(self.train_loader)}] "
                            f"Loss: {loss_meter.avg:.4f} Acc: {running_accuracy:.4f} LR: {lr:.2e}"
                        )

                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {e}")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.global_step += 1
                progress.update(task, advance=1)

        epoch_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        return loss_meter.avg, epoch_accuracy
    
    def _init_data(self):
        """Initialize datasets and dataloaders"""
        data_config = self.config['data']

        console.print("[yellow]Loading datasets...[/yellow]")

        max_seq_length = data_config.get('max_length', 512)
        use_ocr = data_config.get('use_ocr', True)
        batch_size = data_config['batch_size']
        num_workers = data_config.get('preprocessing_num_workers', 4)
        num_labels = self.config['model']['num_labels']

        # Create datasets using OCR-enabled pipeline
        self.train_dataset = LayoutXLMDataset(
            data_config['train'],
            processor=self.processor,
            max_seq_length=max_seq_length,
            use_ocr=use_ocr,
            num_labels=num_labels
        )

        self.val_dataset = LayoutXLMDataset(
            data_config['validation'],
            processor=self.processor,
            max_seq_length=max_seq_length,
            use_ocr=use_ocr,
            num_labels=num_labels
        )

        self.test_dataset = LayoutXLMDataset(
            data_config['test'],
            processor=self.processor,
            max_seq_length=max_seq_length,
            use_ocr=use_ocr,
            num_labels=num_labels
        )

        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn_layoutxlm,
            pin_memory=True
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            **loader_kwargs
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            **loader_kwargs
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            **loader_kwargs
        )

        self.logger.info(
            "Dataset sizes - Train: %d | Val: %d | Test: %d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset)
        )

        console.print(Panel.fit(
            f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)} | Test: {len(self.test_dataset)}",
            title="Dataset Overview"
        ))

    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler"""
        train_config = self.config['training']
        optimizer_config = self.config.get('optimizer', {})
        scheduler_config = self.config.get('scheduler', {})

        learning_rate = train_config['learning_rate']
        weight_decay = train_config.get('weight_decay', 0.0)
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        eps = optimizer_config.get('eps', 1e-8)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )

        num_epochs = train_config['num_epochs']
        grad_accum_steps = train_config.get('gradient_accumulation_steps', 1)
        steps_per_epoch = math.ceil(len(self.train_loader) / grad_accum_steps)
        total_training_steps = steps_per_epoch * num_epochs

        warmup_steps = scheduler_config.get('num_warmup_steps')
        if warmup_steps is None:
            warmup_ratio = train_config.get('warmup_ratio', 0.0)
            warmup_steps = int(total_training_steps * warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps
        )

        self.total_training_steps = total_training_steps

        self.logger.info(
            "Optimizer initialized — LR: %.2e | Weight Decay: %.2e | Warmup Steps: %d | Total Steps: %d",
            learning_rate,
            weight_decay,
            warmup_steps,
            total_training_steps
        )
    
    def validate(self, epoch: int):
        """Validate model"""
        self.model.eval()

        loss_meter = AverageMeter()
        correct_tokens = 0
        total_tokens = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation", leave=False)):
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    bbox = batch['bbox'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    image = batch['pixel_values'].to(self.device)  # LayoutXLM uses 'image' parameter
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        bbox=bbox,
                        image=image,  # LayoutXLM uses 'image' not 'pixel_values'
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss_value = outputs.loss.item()
                    logits = outputs.logits

                    predictions = torch.argmax(logits, dim=-1).detach()
                    valid_mask = labels != -100
                    valid_labels = labels[valid_mask].detach()
                    valid_predictions = predictions[valid_mask]

                    valid_total = valid_labels.numel()
                    if valid_total > 0:
                        correct_tokens += (valid_predictions == valid_labels).sum().item()
                        total_tokens += valid_total
                        loss_meter.update(loss_value, valid_total)
                    else:
                        loss_meter.update(loss_value)

                except Exception as e:
                    self.logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue

        epoch_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

        wandb.log({
            'val/loss': loss_meter.avg,
            'val/accuracy': epoch_accuracy,
            'val/epoch': epoch
        })

        self.logger.info(f"Validation Loss: {loss_meter.avg:.4f} Accuracy: {epoch_accuracy:.4f}")

        return loss_meter.avg, epoch_accuracy
    
    def evaluate_on_test(self, save_report: bool = True):
        """Evaluate model on test set and calculate metrics"""
        console.print(Panel.fit(
            "[bold cyan]Evaluating on Test Set[/bold cyan]",
            title="Test Evaluation"
        ))

        self.logger.info(f"Test samples: {len(self.test_dataset)}")

        self.model.eval()

        loss_meter = AverageMeter()
        correct_tokens = 0
        total_tokens = 0
        y_true: List[int] = []
        y_pred: List[int] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Testing", leave=False)):
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    bbox = batch['bbox'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    image = batch['pixel_values'].to(self.device)  # LayoutXLM uses 'image' parameter
                    labels = batch['labels'].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        bbox=bbox,
                        image=image,  # LayoutXLM uses 'image' not 'pixel_values'
                        attention_mask=attention_mask,
                        labels=labels
                    )

                    loss_value = outputs.loss.item()
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    valid_mask = labels != -100

                    valid_labels = labels[valid_mask]
                    valid_predictions = predictions[valid_mask]
                    valid_total = valid_labels.numel()

                    if valid_total > 0:
                        matches = (valid_predictions == valid_labels).sum().item()
                        correct_tokens += matches
                        total_tokens += valid_total

                        y_true.extend(valid_labels.detach().cpu().tolist())
                        y_pred.extend(valid_predictions.detach().cpu().tolist())
                        loss_meter.update(loss_value, valid_total)
                    else:
                        loss_meter.update(loss_value)

                except Exception as e:
                    self.logger.error(f"Error in test batch {batch_idx}: {e}")
                    continue

        if total_tokens == 0:
            self.logger.warning("No valid tokens found during test evaluation.")
            return {}

        label_ids = sorted(self.id2label.keys())
        target_names = [self.id2label[label_id] for label_id in label_ids]

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=target_names,
            zero_division=0,
            output_dict=True
        )

        accuracy = accuracy_score(y_true, y_pred)
        macro_precision = report_dict.get('macro avg', {}).get('precision', 0.0)
        macro_recall = report_dict.get('macro avg', {}).get('recall', 0.0)
        macro_f1 = report_dict.get('macro avg', {}).get('f1-score', 0.0)

        class_accuracy: List[float] = []
        for label_name in target_names:
            class_metrics = report_dict.get(label_name, {})
            class_accuracy.append(class_metrics.get('recall', 0.0))

        results_table = Table(title="Test Results", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="magenta")

        results_table.add_row("Loss", f"{loss_meter.avg:.4f}")
        results_table.add_row("Accuracy", f"{accuracy:.4f} ({accuracy * 100:.2f}%)")
        results_table.add_row("Macro Precision", f"{macro_precision:.4f}")
        results_table.add_row("Macro Recall", f"{macro_recall:.4f}")
        results_table.add_row("Macro F1", f"{macro_f1:.4f}")
        results_table.add_row("Valid Tokens", str(total_tokens))

        console.print(results_table)

        wandb_logs = {
            'test/loss': loss_meter.avg,
            'test/accuracy': accuracy,
            'test/macro_precision': macro_precision,
            'test/macro_recall': macro_recall,
            'test/macro_f1': macro_f1
        }

        for label_name, class_acc in zip(target_names, class_accuracy):
            wandb_logs[f'test/accuracy_{label_name}'] = class_acc

        wandb.log(wandb_logs)

        self.logger.info(
            "Test results — Loss: %.4f | Accuracy: %.4f | Macro F1: %.4f",
            loss_meter.avg,
            accuracy,
            macro_f1
        )

        if save_report:
            self._save_metrics(report_dict)

        return {
            'test_loss': loss_meter.avg,
            'test_accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'class_accuracy': class_accuracy,
            'total_samples': len(self.test_dataset),
            'correct_predictions': correct_tokens,
            'total_predictions': total_tokens,
            'report': report_dict
        }
    
    def _save_metrics(self, report_dict):
        """Persist classification report to disk"""
        metrics_dir = self.config['checkpoint'].get('save_dir', '.')
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, 'test_classification_report.json')

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2)

        self.logger.info(f"Saved test classification report to {metrics_path}")
    
    def save_model(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_config = self.config['checkpoint']
        save_dir = checkpoint_config['save_dir']
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save regular checkpoint
        if epoch % checkpoint_config.get('save_every_n_epochs', 5) == 0:
            save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                self.global_step,
                self.best_metric,
                save_path,
                self.config
            )
            self.logger.info(f"Saved checkpoint: {save_path}")
            console.print(f"[green]✓ Checkpoint saved: {save_path}[/green]")
        
        # Save best model
        if is_best:
            best_path = os.path.join(save_dir, "best_model.pt")
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                self.global_step,
                self.best_metric,
                best_path,
                self.config
            )
            self.logger.info(f"Saved best model: {best_path}")
            console.print(f"[bold green]★ Best model saved: {best_path}[/bold green]")
    
    def save_training_results(self, test_results=None):
        """Save comprehensive training and test results to file"""
        results_file = "training_results.txt"
        
        # Calculate training duration
        if self.training_start_time and self.training_end_time:
            duration = self.training_end_time - self.training_start_time
            hours, remainder = divmod(int(duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours}h {minutes}m {seconds}s"
        else:
            duration_str = "N/A"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LayoutXLM Document Layout Analysis - Training Results\n")
            f.write("=" * 80 + "\n\n")
            
            # Training Configuration
            f.write("TRAINING CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Model: {self.config['model']['pretrained_model_name']}\n")
            f.write(f"Number of Labels: {self.config['model']['num_labels']}\n")
            f.write(f"Training Epochs: {self.config['training']['num_epochs']}\n")
            f.write(f"Learning Rate: {self.config['training']['learning_rate']}\n")
            f.write(f"Batch Size: {self.config['data']['batch_size']}\n")
            f.write(f"Gradient Accumulation Steps: {self.config['training']['gradient_accumulation_steps']}\n")
            f.write(f"Effective Batch Size: {self.config['data']['batch_size'] * self.config['training']['gradient_accumulation_steps']}\n")
            f.write(f"Weight Decay: {self.config['training']['weight_decay']}\n")
            f.write(f"Warmup Ratio: {self.config['training']['warmup_ratio']}\n")
            f.write(f"Max Gradient Norm: {self.config['training']['max_grad_norm']}\n")
            f.write(f"Random Seed: {self.config['training']['seed']}\n")
            f.write(f"Device: {self.device}\n")
            f.write("\n")
            
            # Training Time
            f.write("TRAINING TIME\n")
            f.write("-" * 80 + "\n")
            if self.training_start_time:
                f.write(f"Start Time: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if self.training_end_time:
                f.write(f"End Time: {self.training_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {duration_str}\n")
            f.write("\n")
            
            # Dataset Information
            f.write("DATASET INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Training Samples: {len(self.train_dataset)}\n")
            f.write(f"Validation Samples: {len(self.val_dataset)}\n")
            if test_results:
                f.write(f"Test Samples: {test_results['total_samples']}\n")
            f.write("\n")
            
            # Model Information
            f.write("MODEL INFORMATION\n")
            f.write("-" * 80 + "\n")
            param_counts = count_parameters(self.model)
            f.write(f"Total Parameters: {param_counts['total']:,}\n")
            f.write(f"Trainable Parameters: {param_counts['trainable']:,}\n")
            f.write(f"Frozen Parameters: {param_counts['frozen']:,}\n")
            f.write("\n")
            
            # Training Results
            f.write("TRAINING RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Best Validation Loss: {self.best_metric:.4f}\n")
            if self.training_history['train_loss']:
                f.write(f"Final Training Loss: {self.training_history['train_loss'][-1]:.4f}\n")
            if self.training_history['train_accuracy']:
                f.write(f"Final Training Accuracy: {self.training_history['train_accuracy'][-1]:.4f} ({self.training_history['train_accuracy'][-1]*100:.2f}%)\n")
            if self.training_history['val_loss']:
                f.write(f"Final Validation Loss: {self.training_history['val_loss'][-1]:.4f}\n")
            if self.training_history['val_accuracy']:
                f.write(f"Final Validation Accuracy: {self.training_history['val_accuracy'][-1]:.4f} ({self.training_history['val_accuracy'][-1]*100:.2f}%)\n")
            f.write(f"Total Training Steps: {self.global_step}\n")
            f.write("\n")
            
            # Test Results (if available)
            if test_results:
                f.write("TEST RESULTS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Test Loss: {test_results['test_loss']:.4f}\n")
                f.write(f"Test Accuracy: {test_results['test_accuracy']:.4f} ({test_results['test_accuracy']*100:.2f}%)\n")
                f.write(f"Correct Predictions: {test_results['correct_predictions']}/{test_results['total_predictions']}\n")
                f.write("\n")
                
                # Per-class accuracy
                f.write("PER-CLASS ACCURACY\n")
                f.write("-" * 80 + "\n")
                id2label = self.config['model'].get('id2label', {})
                for c in range(self.model.config.num_labels):
                    label_name = id2label.get(str(c), f"Class {c}")
                    acc = test_results['class_accuracy'][c]
                    f.write(f"{label_name:15s}: {acc:.4f} ({acc*100:.2f}%)\n")
                f.write("\n")
            
            # Training History (last 10 epochs)
            if self.training_history['train_loss']:
                f.write("TRAINING HISTORY (Last 10 Epochs)\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<12}\n")
                f.write("-" * 80 + "\n")
                
                start_idx = max(0, len(self.training_history['train_loss']) - 10)
                for i in range(start_idx, len(self.training_history['train_loss'])):
                    epoch_num = i + 1
                    train_loss = self.training_history['train_loss'][i]
                    train_acc = self.training_history['train_accuracy'][i] if i < len(self.training_history['train_accuracy']) else 0.0
                    val_loss = self.training_history['val_loss'][i] if i < len(self.training_history['val_loss']) else 0.0
                    val_acc = self.training_history['val_accuracy'][i] if i < len(self.training_history['val_accuracy']) else 0.0
                    lr = self.training_history['learning_rate'][i] if i < len(self.training_history['learning_rate']) else 0.0
                    f.write(f"{epoch_num:<8} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f} {lr:<12.2e}\n")
                f.write("\n")
            
            # Checkpoint Information
            f.write("CHECKPOINT INFORMATION\n")
            f.write("-" * 80 + "\n")
            checkpoint_dir = self.config['checkpoint']['save_dir']
            f.write(f"Checkpoint Directory: {checkpoint_dir}\n")
            if os.path.exists(os.path.join(checkpoint_dir, "best_model.pt")):
                f.write(f"Best Model: {os.path.join(checkpoint_dir, 'best_model.pt')}\n")
            f.write("\n")
            
            # Wandb Information
            f.write("WEIGHTS & BIASES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Project: {self.config['wandb']['project']}\n")
            f.write(f"Run Name: {self.config['wandb']['name']}\n")
            if wandb.run:
                f.write(f"Run URL: {wandb.run.get_url()}\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"Results saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n")
        
        console.print(f"\n[bold green]✓ Results saved to: {results_file}[/bold green]")
        self.logger.info(f"Training results saved to {results_file}")
    
    def train(self):
        """Main training loop"""
        train_config = self.config['training']
        num_epochs = train_config['num_epochs']
        patience = train_config.get('early_stopping_patience')
        patience_counter = 0
        
        console.print(Panel.fit(
            f"[bold cyan]Starting Training for {num_epochs} Epochs[/bold cyan]",
            title="Training"
        ))
        
        # Record start time
        self.training_start_time = datetime.now()
        self.logger.info(f"Training started at: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Check if best model
            is_best = val_loss < self.best_metric
            if is_best:
                self.best_metric = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_model(epoch, is_best)
            
            wandb.log({
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'train/epoch_accuracy': train_acc,
                'val/epoch_loss': val_loss,
                'val/epoch_accuracy': val_acc,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })

            # Epoch summary
            table = Table(title=f"Epoch {epoch} Summary", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Train Loss", f"{train_loss:.4f}")
            table.add_row("Train Accuracy", f"{train_acc:.4f} ({train_acc*100:.2f}%)")
            table.add_row("Val Loss", f"{val_loss:.4f}")
            table.add_row("Val Accuracy", f"{val_acc:.4f} ({val_acc*100:.2f}%)")
            table.add_row("Best Val Loss", f"{self.best_metric:.4f}")
            table.add_row("Learning Rate", f"{self.optimizer.param_groups[0]['lr']:.2e}")
            
            console.print(table)

            if patience and patience_counter >= patience:
                self.logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    patience
                )
                console.print(Panel.fit(
                    f"[bold yellow]Early stopping triggered at epoch {epoch}[/bold yellow]",
                    title="Early Stopping"
                ))
                break
        
        # Record end time
        self.training_end_time = datetime.now()
        duration = self.training_end_time - self.training_start_time
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        console.print(Panel.fit(
            "[bold green]✓ Training Completed Successfully![/bold green]\n"
            f"Total Time: {hours}h {minutes}m {seconds}s",
            title="Complete"
        ))
        
        self.logger.info(f"Training completed at: {self.training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total training time: {hours}h {minutes}m {seconds}s")
        
        # Run evaluation on test set
        console.print("\n")
        test_results = self.evaluate_on_test()
        
        # Save comprehensive results
        self.save_training_results(test_results)
        
        wandb.finish()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train LayoutXLM for Document Layout Analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to configuration YAML file (default: config.yml)"
    )
    args = parser.parse_args()
    
    try:
        console.print(Panel.fit(
            f"[bold cyan]Loading configuration from: {args.config}[/bold cyan]",
            title="Setup"
        ))
        
        trainer = LayoutXLMTrainer(config_path=args.config)
        trainer.train()
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
