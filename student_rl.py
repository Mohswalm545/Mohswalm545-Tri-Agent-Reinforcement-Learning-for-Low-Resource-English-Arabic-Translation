"""
STJ RL Student (A)
Letter-level reinforcement learning student with imitation stabilization.
Learns to output Arabic words one character at a time to match English inputs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from env_rl import RLWordEnv, ARABIC_CHARS, CH2ID, ID2CH, EOS_ID


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# --- simple neural policy (GRU-based) ---
class Policy(nn.Module):
    def __init__(self, en_dim=26, hidden=128):
        super().__init__()
        self.en_proj = nn.Linear(en_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, len(ARABIC_CHARS) + 1)  # +EOS

    def forward(self, en_vec, seq_len=1):
        h0 = torch.tanh(self.en_proj(en_vec)).unsqueeze(0)  # [1, B, H]
        x = h0.permute(1, 0, 2).repeat(1, seq_len, 1)
        out, _ = self.gru(x, h0)
        logits = self.head(out)
        return logits


# --- student agent ---
class RLStudent:
    def __init__(self, max_len=8, lr=1e-3):
        self.env = RLWordEnv(max_len=max_len)
        self.policy = Policy().to(DEVICE)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.baseline = 0.0
        self.baseline_beta = 0.9

    def train_episode(self):
        """Run one RL episode with imitation stabilization."""
        from judge_translation import ENGLISH_WORDS
        loss_fn = nn.CrossEntropyLoss()

        obs = torch.from_numpy(self.env.reset()).float().unsqueeze(0).to(DEVICE)
        logps = []
        done = False
        steps = 0
        reward = 0.0
        info = {}

        while not done and steps < 12:
            logits = self.policy(obs, seq_len=1)[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logps.append(dist.log_prob(action))

            _, reward, done, info = self.env.step(action.item())
            steps += 1

        # --- RL loss ---
        adv = reward - self.baseline
        self.baseline = self.baseline_beta * self.baseline + (1 - self.baseline_beta) * reward
        rl_loss = -adv * torch.stack(logps).sum()

        # --- imitation loss ---
        imitation_loss = torch.tensor(0.0)
        eng = info.get("eng", "")
        correct_arb = ENGLISH_WORDS.get(eng)
        if correct_arb:
            obs_sup = torch.from_numpy(self.env._encode_english(eng)).float().unsqueeze(0).to(DEVICE)
            target_ids = [CH2ID.get(ch, EOS_ID) for ch in correct_arb] + [EOS_ID]
            target = torch.tensor(target_ids).unsqueeze(0).to(DEVICE)
            logits_sup = self.policy(obs_sup, seq_len=len(target_ids))
            logits_sup = logits_sup[0, : len(target_ids), :]
            imitation_loss = loss_fn(logits_sup.reshape(-1, logits_sup.size(-1)), target.reshape(-1))

        # --- combine losses ---
        total_loss = rl_loss + 0.1 * imitation_loss

        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

        return float(reward), info.get("eng", ""), info.get("arb", "")


# --- supervised warmup (optional pretraining) ---
def supervised_warmup(agent, steps=500):
    """Train model to output real Arabic translations directly."""
    from judge_translation import ENGLISH_WORDS
    data = list(ENGLISH_WORDS.items())
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(agent.policy.parameters(), lr=1e-3)

    for i in range(steps):
        eng, correct_arb = data[i % len(data)]
        obs = torch.from_numpy(agent.env._encode_english(eng)).float().unsqueeze(0)
        target_ids = [CH2ID.get(ch, EOS_ID) for ch in correct_arb] + [EOS_ID]
        target = torch.tensor(target_ids).unsqueeze(0)

        logits = agent.policy(obs, seq_len=len(target_ids))
        logits = logits[0, : len(target_ids), :]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"pretrain step {i+1}/{steps}, loss={loss.item():.4f}")


if __name__ == "__main__":
    torch.set_num_threads(2)
    agent = RLStudent(max_len=8, lr=1e-3)

    print(">>> supervised warm-up (pretraining)")
    supervised_warmup(agent, steps=500)

    print("\n>>> switching to RL training")
    avg_R = 0.0
    alpha = 0.1  # smoothing factor for moving average

    for ep in range(200):  # you can raise this later
        R, eng, arb = agent.train_episode()
        avg_R = alpha * R + (1 - alpha) * avg_R

        if (ep + 1) % 5 == 0:
            print(f"ep {ep+1:03d} | R={R:+.2f} | {eng} -> {arb} | avgR={avg_R:+.2f}")

