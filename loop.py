"""
STJ Main Loop
File: stj/loop.py
Purpose:
    Run the interaction loop between:
        Teacher (B) → Student (A) → Judge (C)
    Handles rewards, logging, and adaptive updates.
"""

import json
from student import Student
from teacher import Teacher
from judge_translation import CompositeJudge

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class STJLoop:
    def __init__(self, rounds: int = 20):
        self.rounds = rounds
        self.teacher = Teacher(fake_prob=0.1)
        self.student = Student()
        self.judge = CompositeJudge()
        self.log = []

    def run(self):
        for i in range(1, self.rounds + 1):
            eng = self.teacher.sample_task()
            arb = self.student.translate(eng)

            reward_c, details = self.judge.score(eng, arb)
            self.student.reward(reward_c)

            student_correct = details["translation_correct"]
            self.teacher.update_difficulty(eng, student_correct)
            reward_b = self.teacher.reward(student_correct, valid_task=details["english_valid"])

            record = {
                "round": i,
                "english": eng,
                "arabic": arb,
                "judge_reward": reward_c,
                "teacher_reward": reward_b,
                "details": details,
            }
            self.log.append(record)

            print(f"Round {i}: B→{eng}, A→{arb} | "
                  f"C_reward={reward_c}, B_reward={reward_b}, details={details}")

        print("\n--- Summary ---")
        print(f"Student stats: {self.student.stats}")
        print(f"Teacher stats: {self.teacher.stats}")
        print(f"Word difficulties: {self.teacher.difficulty}")

    def save_log(self, path="logs/stj_loop.jsonl"):
        import os
        os.makedirs("logs", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in self.log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Saved log to {path}")


if __name__ == "__main__":
    loop = STJLoop(rounds=20)
    loop.run()
    loop.save_log()

