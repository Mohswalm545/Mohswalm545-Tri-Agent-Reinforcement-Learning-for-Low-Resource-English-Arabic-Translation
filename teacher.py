"""
STJ Teacher (B) for Phase 1: English→Arabic word challenge (adaptive version)
File: stj/teacher.py
Purpose:
 - Generate one-word English challenges for Student A.
 - Gradually adapt difficulty based on A’s performance.
 - Occasionally introduce fake words to test A’s detection ability.
"""

import random
from typing import Tuple
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# shared vocabulary (should match judge_translation.py)
ENGLISH_WORDS = {
    'book': 'كتاب', 'sun': 'شمس', 'moon': 'قمر', 'sky': 'سماء', 'car': 'سيارة', 'water': 'ماء',
    'tree': 'شجرة', 'house': 'منزل', 'day': 'يوم', 'night': 'ليل', 'fire': 'نار'
}


class Teacher:
    def __init__(self, fake_prob: float = 0.1):
        self.words = list(ENGLISH_WORDS.keys())
        self.difficulty = {w: 1.0 for w in self.words}  # adaptive weights
        self.fake_prob = fake_prob
        self.last_word = None
        self.stats = {'wins': 0, 'losses': 0, 'fake_used': 0}

    def _make_fake_word(self) -> str:
        """Generate a random fake English-looking word."""
        length = random.randint(3, 6)
        return ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length))

    def sample_task(self) -> str:
        """Pick one English word (weighted by difficulty) or a fake word."""
        # maybe generate fake
        if random.random() < self.fake_prob:
            fake = self._make_fake_word()
            self.last_word = fake
            self.stats['fake_used'] += 1
            return fake

        # weighted random choice
        weights = [self.difficulty[w] for w in self.words]
        word = random.choices(self.words, weights=weights, k=1)[0]
        self.last_word = word
        return word

    def update_difficulty(self, word: str, student_correct: bool):
        """Adjust difficulty based on A's success."""
        if word not in self.difficulty:
            return
        if student_correct:
            self.difficulty[word] *= 0.8  # reduce chance
        else:
            self.difficulty[word] *= 1.2  # increase chance
        # keep weights in reasonable range
        self.difficulty[word] = max(0.1, min(5.0, self.difficulty[word]))

    def reward(self, student_correct: bool, valid_task: bool = True) -> int:
        """Teacher’s reward:
        +1 if A got it wrong (Teacher wins)
         0 if A got it right (tie)
        -1 if Teacher gave invalid/garbage input (penalty)
        """
        if not valid_task:
            r = -1
        elif student_correct:
            r = 0
        else:
            r = 1

        if r == 1:
            self.stats['wins'] += 1
        elif r == 0:
            self.stats['losses'] += 1
        return r


if __name__ == '__main__':
    from judge_translation import CompositeJudge

    B = Teacher(fake_prob=0.2)
    C = CompositeJudge()

    FAKE_STUDENT_ANSWERS = {
        'book': 'كتاب',  # correct
        'sun': 'قمر',    # wrong
        'sky': 'سماء',   # correct
        'tree': 'شمس',   # wrong
        'fire': 'نار'    # correct
    }

    for i in range(10):
        eng = B.sample_task()
        arb = FAKE_STUDENT_ANSWERS.get(eng, 'خطأ')
        reward_c, details = C.score(eng, arb)
        student_correct = details['translation_correct']
        B.update_difficulty(eng, student_correct)
        reward_b = B.reward(student_correct, valid_task=details['english_valid'])

        print(f"Round {i+1}: B→{eng}, A→{arb} | C_reward={reward_c}, B_reward={reward_b}, details={details}")

    print(f"Teacher stats: {B.stats}")
    print(f"Word difficulties: {B.difficulty}")

