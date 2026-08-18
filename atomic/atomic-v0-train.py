import mmap
import random
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as func

import pickle

parser = argparse.ArgumentParser(description='Train the Atomic GPT model (v0) on data.')

parser.add_argument('--block_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()
print(f'block size: {args.block_size}, batch size: {args.batch_size}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BLOCK_SIZE = args.block_size
BATCH_SIZE = args.batch_size

MAX_ITERATIONS = 3000
LEARN_RATE = 3e-4
EVAL_INTERVAL = 100

N_EMBED = 384
N_HEAD = 4
N_LAYER = 4

DROPOUT = 0.1

print(device)

vocab = ""
with open("openwebtext/vocab.txt", "r", encoding='utf-8') as vocab_file:
    text = vocab_file.read()
    vocab = sorted(list(set(text)))

vocab_len = len(vocab)

string_to_int = { ch:i for i, ch in enumerate(vocab) }
int_to_string = { i:ch for i, ch in enumerate(vocab) }

encode = lambda s: [string_to_int[c] for c in s ]
decode = lambda l: ''.join([int_to_string[i] for i in l])

def get_random_chunk(split):
    filename = "openwebtext/output_train.txt" if split == 'train' else "openwebtext/output_valid.txt"

    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mem_map:
            file_size = len(mem_map)
            start_pos = random.randint(0, (file_size) - BLOCK_SIZE * BATCH_SIZE)

            mem_map.seek(start_pos)

            block = mem_map.read(BLOCK_SIZE * BATCH_SIZE - 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')

            data = torch.tensor(encode(decoded_block), dtype=torch.long)

    return data

def get_batch(split):
    data = get_random_chunk(split)

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'validate']:
        losses = torch.zeros(EVAL_INTERVAL)

        for k in range(EVAL_INTERVAL):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
        
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        BATCH, TIME, CHANNEL = x.shape
        
        key = self.key(x)
        query = self.query(x)

        weights = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        weights = weights.masked_fill(self.tril[:TIME, :TIME] == 0, float('-inf'))
        weights = func.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        value = self.value(x)
        out = weights @ value

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.project = nn.Linear(head_size * n_head, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.project(out))

        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()

        head_size = n_embed // n_head

        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_fwd = FeedForward(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        y = self.self_attention(x)
        x = self.layer_norm1(x + y)
        y = self.feed_fwd(x)
        x = self.layer_norm2(x + y)

        return x

class GPTLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embed_table = nn.Embedding(vocab_size, N_EMBED)
        self.pos_embed_table   = nn.Embedding(BLOCK_SIZE, N_EMBED)

        self.blocks = nn.Sequential(*[Block(N_EMBED, n_head=N_HEAD) for _ in range(N_LAYER)])

        self.layer_norm_final = nn.LayerNorm(N_EMBED)
        self.lang_model_head = nn.Linear(N_EMBED, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.2)

    def forward(self, index, targets=None):
        BATCH, TIME = index.shape

        token_embed = self.token_embed_table(index)
        pos_embed = self.pos_embed_table(torch.arange(TIME, device=device))

        x = token_embed + pos_embed
        x = self.blocks(x)
        x = self.layer_norm_final(x)

        logits = self.lang_model_head(x)

        if targets == None:
            loss = None
        else:
            BATCH, TIME, CHANNEL = logits.shape
            logits = logits.view(BATCH * TIME, CHANNEL)
            targets = targets.view(BATCH * TIME)
            loss = func.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -BLOCK_SIZE:]
            logits, loss = self.forward(index_cond)

            logits = logits[:, -1, :]
            probs = func.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)

        return index

model = GPTLM(vocab_len)

print('loading model params...')
with open('model-01.pkl', 'rb') as f:
    model = pickle.load(f)
print('loaded successfully!')

m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARN_RATE)

for iteration in range(MAX_ITERATIONS):
    if iteration % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {iteration:4d} | train: {losses['train']:.4f} | val: {losses['validate']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model.forward(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)

print('model saved')
