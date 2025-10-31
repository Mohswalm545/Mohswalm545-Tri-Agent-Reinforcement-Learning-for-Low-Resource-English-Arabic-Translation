"""
Environment for Phase-1 English→Arabic 1-word translation.
- Observation: encoding of the English word (bag of letters).
- Action space: Arabic letters + <eos> (end of sequence).
- Episode ends on <eos> or when max length is reached.
- Reward: shaped based on closeness to correct Arabic translation.
"""

from typing import Tuple
import numpy as np
from judge_translation import CompositeJudge
from teacher import Teacher

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Arabic characters allowed for generation
ARABIC_CHARS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهويأةىآؤئ") + [" "]
ID2CH = {i: ch for i, ch in enumerate(ARABIC_CHARS + ["<eos>"])}
CH2ID = {ch: i for i, ch in ID2CH.items()}
EOS_ID = CH2ID["<eos>"]

# English alphabet
EN_CHARS = list("abcdefghijklmnopqrstuvwxyz")
EN2ID = {c: i for i, c in enumerate(EN_CHARS)}


class RLWordEnv:
    def __init__(self, max_len: int = 8):
        self.max_len = max_len
        self.teacher = Teacher(fake_prob=0.1)
        self.judge = CompositeJudge()
        self.cur_eng = None
        self.decoded = []

    def _encode_english(self, word: str) -> np.ndarray:
        """Convert English word to simple bag-of-letters vector."""
        v = np.zeros(len(EN_CHARS), dtype=np.float32)
        for c in word.lower():
            if c in EN2ID:
                v[EN2ID[c]] += 1.0
        s = v.sum()
        return v / s if s > 0 else v

    def reset(self) -> np.ndarray:
        """Start a new environment episode."""
        self.cur_eng = self.teacher.sample_task()
        self.decoded = []
        return self._encode_english(self.cur_eng)

    def step(self, action_id: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Perform one step (generate one Arabic character)."""
        done = False
        reward = 0.0
        info = {}

        if action_id == EOS_ID or len(self.decoded) >= self.max_len:
            done = True
            arb = "".join(self.decoded) if self.decoded else ""
            r_c, details = self.judge.score(self.cur_eng, arb)

            # --- reward shaping ---
            reward = float(r_c)  # base from judge

            if not details["english_valid"]:
                reward -= 0.5  # penalty for fake English
            else:
                # get ground truth if available
                from judge_translation import ENGLISH_WORDS
                true_word = ENGLISH_WORDS.get(self.cur_eng)
                if true_word:
                    # partial letter overlap reward
                    match_letters = sum(1 for a, b in zip(arb, true_word) if a == b)
                    partial = match_letters / max(len(true_word), 1)

                    # combine partial + correctness
                    if arb == true_word:
                        reward = 1.0
                    elif partial > 0.5:
                        reward = 0.5
                    elif partial > 0.25:
                        reward = 0.25
                    else:
                        reward = reward - 0.5 if reward > 0 else -0.25

            # --- teacher updates ---
            student_correct = details["translation_correct"]
            self.teacher.update_difficulty(self.cur_eng, student_correct)
            self.teacher.reward(student_correct, valid_task=details["english_valid"])

            info = {"eng": self.cur_eng, "arb": arb, "details": details}

        else:
            ch = ID2CH.get(action_id, " ")
            if ch != "<eos>":
                self.decoded.append(ch)

        obs = self._encode_english(self.cur_eng)
        return obs, reward, done, info

