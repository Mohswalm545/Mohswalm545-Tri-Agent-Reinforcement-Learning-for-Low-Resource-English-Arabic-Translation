"""
STJ Translation Judges (EnglishJudge, ArabicJudge, TranslationJudge, CompositeJudge)
File: stj/judge_translation.py
Purpose: Provide simple rule-based judges for phase‑1 of STJ (1‑word translation phase).
"""

import re
from typing import Dict, Tuple

# --- helpers ---
ENGLISH_WORDS = {
    'book': 'كتاب', 'sun': 'شمس', 'moon': 'قمر', 'sky': 'سماء', 'car': 'سيارة', 'water': 'ماء',
    'tree': 'شجرة', 'house': 'منزل', 'day': 'يوم', 'night': 'ليل', 'fire': 'نار'
}
ARABIC_WORDS = set(ENGLISH_WORDS.values())

# --- Judges ---
class EnglishJudge:
    def check(self, word: str) -> bool:
        return word.lower() in ENGLISH_WORDS

class ArabicJudge:
    def check(self, word: str) -> bool:
        # must contain Arabic letters only (basic regex)
        return bool(re.fullmatch(r'[\u0600-\u06FF]+', word)) and word in ARABIC_WORDS

class TranslationJudge:
    def check(self, eng: str, arb: str) -> bool:
        correct = ENGLISH_WORDS.get(eng.lower())
        return correct == arb

# --- Composite ---
class CompositeJudge:
    def __init__(self):
        self.eng_judge = EnglishJudge()
        self.arb_judge = ArabicJudge()
        self.trans_judge = TranslationJudge()

    def score(self, eng_word: str, arb_word: str) -> Tuple[int, Dict[str, bool]]:
        """Return (reward, details)"""
        e_ok = self.eng_judge.check(eng_word)
        a_ok = self.arb_judge.check(arb_word)
        t_ok = self.trans_judge.check(eng_word, arb_word)
        details = {'english_valid': e_ok, 'arabic_valid': a_ok, 'translation_correct': t_ok}

        # scoring rule
        if e_ok and a_ok and t_ok:
            reward = 1
        elif (e_ok and a_ok) or (t_ok and a_ok):
            reward = 0
        else:
            reward = -1
        return reward, details

if __name__ == '__main__':
    judge = CompositeJudge()
    examples = [
        ('book', 'كتاب'),
        ('sun', 'قمر'),
        ('sky', 'سماء'),
        ('moon', 'قمر'),
        ('fire', 'ماء'),
        ('asdf', 'كتاب'),
        ('car', 'سيارة')
    ]
    for eng, arb in examples:
        r, d = judge.score(eng, arb)
        print(f"{eng} -> {arb}: reward={r}, details={d}")

