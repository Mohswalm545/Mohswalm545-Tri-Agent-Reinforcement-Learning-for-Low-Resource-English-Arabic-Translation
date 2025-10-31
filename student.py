"""
STJ Student (A)
File: stj/student.py
Purpose: Student receives an English word, tries to produce the Arabic translation,
and tracks its own reward stats.
"""

import random
from typing import Tuple
from judge_translation import CompositeJudge
from teacher import Teacher

# shared bilingual dictionary
ENGLISH_WORDS = {
    'book': 'كتاب', 'sun': 'شمس', 'moon': 'قمر', 'sky': 'سماء', 'car': 'سيارة',
    'water': 'ماء', 'tree': 'شجرة', 'house': 'منزل', 'day': 'يوم',
    'night': 'ليل', 'fire': 'نار'
}


class Student:
    def __init__(self):
        self.known_words = dict(ENGLISH_WORDS)  # can learn/add more later
        self.stats = {'correct': 0, 'wrong': 0, 'total': 0}

    def translate(self, english_word: str) -> str:
        """Return the Arabic translation if known, else a random guess."""
        if english_word in self.known_words:
            return self.known_words[english_word]
        else:
            # wrong guess on purpose
            return random.choice(list(self.known_words.values()))

    def reward(self, reward_value: int):
        """Update internal stats."""
        self.stats['total'] += 1
        if reward_value > 0:
            self.stats['correct'] += 1
        elif reward_value < 0:
            self.stats['wrong'] += 1


if __name__ == "__main__":
    B = Teacher()
    A = Student()
    C = CompositeJudge()

    for i in range(5):
        eng = B.sample_task()
        arb = A.translate(eng)
        reward_c, details = C.score(eng, arb)
        A.reward(reward_c)
        student_correct = details['translation_correct']
        reward_b = B.reward(student_correct, valid_task=details['english_valid'])

        print(f"Round {i+1}: B→{eng}, A→{arb} | C_reward={reward_c}, B_reward={reward_b}, details={details}")

    print(f"Student stats: {A.stats}")
    print(f"Teacher stats: {B.stats}")

