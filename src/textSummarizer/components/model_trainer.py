import os
import torch
import gc
from textSummarizer.entity import ModelTrainerConfig
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Clear GPU memory and check availability
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("GPU not available, using CPU (will be slower)")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load tokenizer and model with optimizations
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_t5 = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(device)
        
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_t5)
        
        # Loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        
        # Re-tokenize with T5 tokenizer to fix vocabulary mismatch
        def retokenize_function(examples):
            # Add T5 task prefix
            inputs = ["summarize: " + doc for doc in examples["dialogue"]]
            model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=False)
            
            # Tokenize targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(examples["summary"], max_length=128, truncation=True, padding=False)
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Re-tokenize the dataset and remove problematic columns
        print("Re-tokenizing dataset with T5 tokenizer...")
        dataset_samsum_pt = dataset_samsum_pt.map(
            retokenize_function, 
            batched=True,
            remove_columns=dataset_samsum_pt["train"].column_names  # Remove all original columns
        )
        
        # Use smaller dataset for faster training (remove these lines for full dataset)
        print("Using subset of data for faster training...")
        train_dataset = dataset_samsum_pt["train"].select(range(2000))  # Use 2000 samples instead of ~14k
        eval_dataset = dataset_samsum_pt["validation"].select(range(400))  # Use 400 samples instead of ~800
        
        print(f"Training on {len(train_dataset)} samples")
        print(f"Evaluating on {len(eval_dataset)} samples")
        
        # Optimized training arguments
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=1,  # Reduced from 3 to 1 for testing
            warmup_steps=50,     # Reduced from 500
            per_device_train_batch_size=4,  # Increased from 2
            per_device_eval_batch_size=4,   # Increased from 2
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy='steps',
            eval_steps=50,       # Reduced from 500
            save_steps=100,      # Reduced from 1000
            gradient_accumulation_steps=2,  # Reduced from 8
            fp16=True,           # Enable mixed precision for speed
            dataloader_pin_memory=True,     # Enable for GPU
            dataloader_num_workers=0,       # Set to 0 to avoid multiprocessing issues
            max_steps=500,       # Limit total steps for testing
            save_total_limit=2,  # Keep only 2 checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to=[],        # Disable wandb
            remove_unused_columns=True,  # Remove unused columns automatically
        )
        
        # Create trainer
        trainer = Trainer(
            model=model_t5, 
            args=trainer_args,
            tokenizer=tokenizer, 
            data_collator=seq2seq_data_collator,
            train_dataset=train_dataset,  # Use smaller dataset
            eval_dataset=eval_dataset     # Use smaller dataset
        )
        
        print("Starting training...")
        trainer.train()
        
        print("Training completed! Saving model...")
        
        # Save model and tokenizer
        model_t5.save_pretrained(os.path.join(self.config.root_dir, "t5-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
        
        print(f"Model saved to: {os.path.join(self.config.root_dir, 't5-samsum-model')}")
        print(f"Tokenizer saved to: {os.path.join(self.config.root_dir, 'tokenizer')}")
        
        # Clear GPU memory after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()