import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import gradio as gr

# ====================== MODEL ARCHITECTURE ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, d_ff=512, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        src = self.pos_encoder(self.embedding(src))
        src = src.transpose(0, 1)  # Transformer expects (seq_len, batch, features)
        for layer in self.encoder_layers:
            src = layer(src)
        return self.fc(src.transpose(0, 1))

# ====================== DATA HANDLING ======================
class ChatDataset(Dataset):
    def __init__(self, tokenizer, conversations, max_length=50):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []

        for conv in conversations:
            if len(conv) >= 2:
                input_text = " [SEP] ".join(conv[:-1])
                target_text = conv[-1]
                self.pairs.append((input_text, target_text))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_enc = self.tokenizer(
            src,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tgt_enc = self.tokenizer(
            tgt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": src_enc.input_ids.squeeze(0),
            "labels": tgt_enc.input_ids.squeeze(0)
        }

# ====================== TRAINING ======================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'sep_token': '[SEP]'
    })

    # Sample data
    conversations = [
        ["Hello!", "Hi there! How can I help you today?"],
        ["What's your name?", "I'm a friendly chatbot!"],
        ["Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"],
        ["Goodbye", "See you later!"]
    ]

    # Create dataset
    dataset = ChatDataset(tokenizer, conversations)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Initialize model with correct vocab size
    model = TransformerChatbot(len(tokenizer)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Training loop
    for epoch in range(3):  # Reduced for demo
        model.train()
        total_loss = 0
        for batch in dataloader:
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)

            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    return model, tokenizer

# ====================== GENERATION ======================
def generate_response(model, tokenizer, prompt, max_length=50, temperature=0.7):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)

        logits = outputs[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=-1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# ====================== GRADIO INTERFACE ======================
def create_interface(model, tokenizer):
    def respond(message, history):
        response = generate_response(model, tokenizer, message)
        return response

    interface = gr.ChatInterface(
        respond,
        title="Transformer Chatbot",
        examples=["Hello!", "Tell me a joke", "What's your name?"]
    )
    return interface

if __name__ == "__main__":
    model, tokenizer = train()
    interface = create_interface(model, tokenizer)
    interface.launch(debug=True)
