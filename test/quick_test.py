#!/usr/bin/env python
"""
Quick test training với 1 epoch
Sửa config on-the-fly không cần tạo file mới
"""

import sys
import wandb
from train import LayoutXLMTrainer
from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    console.print(Panel.fit(
        "[bold yellow]Quick Test Training - 1 Epoch Only[/bold yellow]\n"
        "Sẽ test xem training loop có chạy được không",
        title="Test Mode"
    ))
    
    # Load trainer với config gốc
    console.print("\n[cyan]Loading trainer...[/cyan]")
    trainer = LayoutXLMTrainer(config_path="config.yml")
    
    # Override config để test nhanh
    console.print("[cyan]Modifying config for quick test:[/cyan]")
    trainer.config['training']['num_epochs'] = 1
    trainer.config['data']['batch_size'] = 2
    trainer.config['logging']['log_steps'] = 10
    trainer.config['checkpoint']['save_every_n_epochs'] = 1
    trainer.config['wandb']['project'] = 'doclayout-test'
    trainer.config['wandb']['name'] = 'layoutxlm-quicktest'
    
    console.print("  ✓ num_epochs = 1")
    console.print("  ✓ batch_size = 2")
    console.print("  ✓ log_steps = 10")
    console.print("  ✓ wandb project = doclayout-test")
    
    # Reload data với batch size mới
    console.print("\n[cyan]Reloading data with new batch size...[/cyan]")
    trainer._init_data()
    trainer._init_optimizer()

    if wandb.run:
        wandb.config.update(trainer.config, allow_val_change=True)

    console.print(f"  ✓ Train batches: {len(trainer.train_loader)}")
    console.print(f"  ✓ Val batches: {len(trainer.val_loader)}")
    
    # Start training
    console.print("\n[bold green]Starting quick test training...[/bold green]\n")
    
    try:
        trainer.train()
        
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold green]✓ Quick test PASSED![/bold green]\n"
            "Training loop hoạt động bình thường.\n"
            "Có thể chạy full training với:\n"
            "  [cyan]python train.py[/cyan]",
            title="Success"
        ))
        console.print("="*60)
        
    except Exception as e:
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            f"[bold red]✗ Quick test FAILED![/bold red]\n"
            f"Error: {str(e)}\n\n"
            "Vui lòng check lỗi ở trên.",
            title="Error"
        ))
        console.print("="*60)
        raise

if __name__ == "__main__":
    main()
