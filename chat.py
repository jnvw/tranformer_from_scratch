import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from torchmetrics.text import BLEUScore, Perplexity
import gradio as gr

# ====================== MODEL ARCHITECTURE ======================
# [Keep all your model classes exactly the same as in your original code]
# PositionalEncoding, MultiHeadAttention, FeedForward, EncoderLayer, DecoderLayer, TransformerChatbot

# ====================== DATA HANDLING ======================
def load_conversation_data():
    try:
        # Try loading with caching disabled first
        dataset = load_dataset("daily_dialog", cache_dir="./cache", verification_mode='no_checks')

        # Alternative if still failing: use streaming
        # dataset = load_dataset("daily_dialog", streaming=True)

        conversations = []
        for dialog in dataset["train"]["dialog"]:
            if len(dialog) >= 2:
                for i in range(1, len(dialog)):
                    conversations.append(dialog[:i+1])
        return conversations
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to manual dataset
        return [
            ["Hi there!", "Hello! How can I help you today?"],
            ["What's the weather like?", "I'm sorry, I don't have weather information."],
            ["Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"]
        ]

class ConversationDataset(Dataset):
    def __init__(self, tokenizer, conversations, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for conv in conversations:
            try:
                if len(conv) < 2:
                    continue

                input_text = " ".join(conv[:-1])
                target_text = conv[-1]

                input_enc = tokenizer(
                    input_text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                target_enc = tokenizer(
                    target_text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                self.data.append({
                    "input_ids": input_enc.input_ids.squeeze(0),
                    "attention_mask": input_enc.attention_mask.squeeze(0),
                    "labels": target_enc.input_ids.squeeze(0)
                })
            except Exception as e:
                print(f"Skipping malformed conversation: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ====================== TRAINING UTILITIES ======================
# [Keep your train_epoch, evaluate functions exactly the same]

# ====================== GENERATION & DEPLOYMENT ======================
# [Keep your generate_response and create_chat_interface functions the same]

# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    # Configuration
    config = {
        "d_model": 256,
        "num_heads": 8,
        "d_ff": 512,
        "num_layers": 3,
        "max_len": 128,
        "batch_size": 32,
        "lr": 3e-4,
        "epochs": 10,  # Reduced for testing
        "grad_clip": 1.0
    }

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer with error handling
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir="./token_cache")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        # Fallback to basic tokenizer
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Data loading with multiple fallbacks
    conversations = load_conversation_data()
    print(f"Loaded {len(conversations)} conversations")

    dataset = ConversationDataset(tokenizer, conversations, config["max_len"])
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Model initialization
    model = TransformerChatbot(
        vocab_size=tokenizer.vocab_size,
        **{k: config[k] for k in ["d_model", "num_heads", "d_ff", "num_layers", "max_len"]}
    ).to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Training loop
    try:
        for epoch in range(config["epochs"]):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config["grad_clip"])
            print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {train_loss:.4f}")

            # Save checkpoint
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "tokenizer": tokenizer
            }, f"chatbot_checkpoint_epoch{epoch}.pth")
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")

    # Launch interface
    try:
        interface = create_chat_interface(model, tokenizer)
        interface.launch()
    except Exception as e:
        print(f"Interface error: {e}")
        print("Try running the interface separately after saving the model")
