import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CTC(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes + 1)  # +1 for blank token

    def forward(self, x, text=None, is_train=None, batch_max_length=None):
        return self.fc(x)


class Attention(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.attention_cell = AttentionCell(input_dim, hidden_dim, num_classes)
        self.generator = nn.Linear(hidden_dim, num_classes)
        # Initialize weights
        nn.init.xavier_uniform_(self.generator.weight)
        if self.generator.bias is not None:
            nn.init.constant_(self.generator.bias, 0)

    def _char_to_onehot(self, input_char, onehot_dim=None):
        if onehot_dim is None:
            onehot_dim = self.num_classes
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        onehot = torch.zeros(batch_size, onehot_dim, dtype=torch.float32, device=device)
        onehot = onehot.scatter_(1, input_char, 1)
        return onehot

    def forward(self, batch_H, batch_max_length, text=None, is_train=True):
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1

        output_hiddens = torch.zeros(batch_size, num_steps, self.hidden_dim, dtype=torch.float32, device=device)
        hidden = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=device)

        if is_train:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(text[:, i])
                hidden = self.attention_cell(hidden, batch_H, char_onehots)
                output_hiddens[:, i, :] = hidden
            logits = self.generator(output_hiddens.view(-1, self.hidden_dim))
            probs = logits.view(batch_size, num_steps, -1)  # Let the loss function handle the softmax

        else:
            # Inference mode or when text is not provided
            probs = torch.zeros(batch_size, num_steps, self.num_classes, dtype=torch.float32, device=device)
            target = torch.zeros(batch_size, dtype=torch.long, device=device)
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(target)
                hidden = self.attention_cell(hidden, batch_H, char_onehots)
                probs_step = self.generator(hidden)
                probs[:, i, :] = probs_step
                _, target = probs_step.max(dim=1)

        return probs  # (batch_size, num_steps, num_classes)


class AttentionCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.i2h = nn.Linear(input_dim, hidden_dim, bias=False)
        self.h2h = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self.rnn = nn.GRUCell(input_dim + output_dim, hidden_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.i2h.weight)
        nn.init.xavier_uniform_(self.h2h.weight)
        if self.h2h.bias is not None:
            nn.init.constant_(self.h2h.bias, 0)
        nn.init.xavier_uniform_(self.score.weight)

        # Initialize GRU weights
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, prev_hidden, batch_H, char_onehots):
        # [batch_size, num_steps, input_dim] -> [batch_size, num_steps, hidden_dim]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden).unsqueeze(1)

        # Scaled dot-product attention
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))

        # Apply attention with temperature scaling
        alpha = F.softmax(e, dim=1)  # Equation 5, batch_size x num_steps x 1
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # Equation 3, batch_size x input_dim

        # Concatenate context with character embedding
        concat_context = torch.cat([context, char_onehots], dim=1)  # batch_size x (input_dim + output_dim)

        # Update hidden state with GRU
        hidden = self.rnn(concat_context, prev_hidden)

        return hidden


# Postprocessing: Greedy, Beam Search
