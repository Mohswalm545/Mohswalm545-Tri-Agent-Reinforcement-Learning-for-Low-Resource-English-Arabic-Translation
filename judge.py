"""
STJ Judge module (frozen strong model)
File: stj/judge.py
Purpose: Provide a callable Judge class that wraps a pretrained transformer-based reward model
and scores (prompt, answer) pairs into {-1, 0, +1} based on a configurable rubric.

Usage:
  - Place your pretrained judge model (PyTorch HF or local path) and set MODEL_NAME or pass to
    Judge.load_model(...).
  - The Judge is frozen by default (no training methods). It includes utilities to batch-score
    examples and to produce a granular score (float) plus the discrete {-1,0,1} label.

Notes:
  - This file intentionally avoids any self-update logic (per your request). If later you want
    to enable updates, we can add distillation/DPO code in a separate file.
"""

from typing import List, Tuple, Dict, Optional
import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Rubric:
    """Simple weighted rubric for Judge.
    R = w_m*meaning + w_g*grammar + w_f*fluency + w_t*task
    Each component should be scored in [-1.0, 1.0]. The final R is normalized and
    thresholded into {-1,0,1}.
    """

    def __init__(self, w_m=0.4, w_g=0.3, w_t=0.2, w_f=0.1,
                 thresh_pos=0.2, thresh_neg=-0.2):
        self.w_m = w_m
        self.w_g = w_g
        self.w_t = w_t
        self.w_f = w_f
        self.thresh_pos = thresh_pos
        self.thresh_neg = thresh_neg

    def score_components(self, comps: Dict[str, float]) -> float:
        """Combine component scores into a single scalar in [-1, 1].
        comps keys: 'meaning', 'grammar', 'task', 'fluency'
        """
        m = comps.get('meaning', 0.0)
        g = comps.get('grammar', 0.0)
        t = comps.get('task', 0.0)
        f = comps.get('fluency', 0.0)
        R = self.w_m * m + self.w_g * g + self.w_t * t + self.w_f * f
        # clamp
        R = max(-1.0, min(1.0, R))
        return R

    def discretize(self, r: float) -> int:
        if r >= self.thresh_pos:
            return 1
        if r <= self.thresh_neg:
            return -1
        return 0


class Judge:
    """Judge wrapper that uses a pretrained sequence classification model
    to produce component-wise and final scores.

    Expected model: takes pair (prompt \t answer) or concatenated input and returns logits
    representing a scalar quality score. If you have a model trained to predict reward
    directly, set `model_is_reward=True` and the code will normalize logits to [-1,1].
    """

    def __init__(self, model_name_or_path: str = None, device: str = None, model_is_reward: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name_or_path
        self.model_is_reward = model_is_reward
        self.tokenizer = None
        self.model = None
        self.rubric = Rubric()

    def load_model(self, model_name_or_path: Optional[str] = None):
        name = model_name_or_path or self.model_name
        if name is None:
            raise ValueError("Provide a HuggingFace model name or local path to load the judge model.")
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.model.to(self.device)
        self.model.eval()
        self.model_name = name

    def _prepare_inputs(self, prompts: List[str], answers: List[str], max_length: int = 512):
        pair_texts = [f"Prompt: {p} \nAnswer: {a}" for p, a in zip(prompts, answers)]
        return self.tokenizer(pair_texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')

    def _model_score(self, prompts: List[str], answers: List[str]) -> List[float]:
        inputs = self._prepare_inputs(prompts, answers)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits.squeeze(-1)
            # If model returns multi-dim logits (num_labels>1), reduce to scalar by difference
            if logits.dim() > 1:
                # try: logits[:, 1] - logits[:, 0]
                logits = logits[:, 0]
            scores = logits.cpu().tolist()
        # normalize to [-1,1] using tanh (if reward) or linear scaling can also be used
        if self.model_is_reward:
            scores = [math.tanh(s) for s in scores]
        else:
            # attempt to scale probabilistically via sigmoid then map to [-1,1]
            scores = [(2.0 * (1 / (1 + math.exp(-s))) - 1.0) for s in scores]
        return scores

    def score(self, prompt: str, answer: str) -> Tuple[float, int, Dict[str, float]]:
        """Score a single pair. Returns (R_float, R_disc, components_dict).
        For a frozen Judge we use model scalar as a proxy for "meaning" and "fluency" and
        simple heuristics for grammar/task.
        """
        # Use model as primary signal
        model_score = self._model_score([prompt], [answer])[0]
        # Heuristic components (placeholders — can be improved with specialized modules)
        comps = {}
        comps['meaning'] = model_score  # model acts as semantic adequacy proxy
        comps['fluency'] = max(-1.0, min(1.0, 1.0 - (self._count_nonletters(answer) / max(1, len(answer.split())))))
        comps['grammar'] = self._grammar_signal(answer)
        comps['task'] = self._task_signal(prompt, answer)
        R_float = self.rubric.score_components(comps)
        R_disc = self.rubric.discretize(R_float)
        return R_float, R_disc, comps

    def batch_score(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[Tuple[float, int, Dict[str, float]]]:
        results = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i+batch_size]
            prompts, answers = zip(*chunk)
            model_scores = self._model_score(list(prompts), list(answers))
            for ms, p, a in zip(model_scores, prompts, answers):
                comps = {}
                comps['meaning'] = ms
                comps['fluency'] = max(-1.0, min(1.0, 1.0 - (self._count_nonletters(a) / max(1, len(a.split())))))
                comps['grammar'] = self._grammar_signal(a)
                comps['task'] = self._task_signal(p, a)
                R_float = self.rubric.score_components(comps)
                R_disc = self.rubric.discretize(R_float)
                results.append((R_float, R_disc, comps))
        return results

    @staticmethod
    def _count_nonletters(text: str) -> int:
        # crude: count chars that are not in Arabic/Latin letters or spaces
        return sum(1 for ch in text if not (ch.isalpha() or ch.isspace()))

    @staticmethod
    def _grammar_signal(text: str) -> float:
        # Placeholder heuristic for grammar: penalize repeated punctuation, long runs of same char,
        # and extreme token repetition. Returns in [-1,1].
        if not text.strip():
            return -1.0
        if ".." in text or "!!!" in text:
            return -0.5
        words = text.split()
        if len(words) < 2:
            return 0.0
        # repetition penalty
        unique = len(set(words)) / len(words)
        return max(-1.0, min(1.0, (unique - 0.5) * 2))

    @staticmethod
    def _task_signal(prompt: str, answer: str) -> float:
        # Simple check: overlap of key tokens from prompt in answer (placeholder).
        p_tokens = set(prompt.split())
        a_tokens = set(answer.split())
        if not p_tokens:
            return 0.0
        overlap = len(p_tokens & a_tokens) / len(p_tokens)
        return max(-1.0, min(1.0, (overlap - 0.2) / 0.8))


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='HF name or local path of judge model')
    parser.add_argument('--input', type=str, default=None, help='JSONL file with {"prompt":"..","answer":".."} per line')
    parser.add_argument('--out', type=str, default='judge_scores.jsonl')
    args = parser.parse_args()

    judge = Judge(model_name_or_path=args.model)
    if args.model:
        judge.load_model(args.model)
    else:
        print('WARNING: no model provided. Please pass --model <hf-name-or-path>. Exiting.')
        raise SystemExit(1)

    if not args.input:
        print('No input file provided. Example usage: --input eval_pairs.jsonl')
        raise SystemExit(1)

    out_f = open(args.out, 'w', encoding='utf-8')
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            p = data.get('prompt','')
            a = data.get('answer','')
            r_float, r_disc, comps = judge.score(p, a)
            rec = {'prompt': p, 'answer': a, 'r_float': r_float, 'r_disc': r_disc, 'components': comps}
            out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    out_f.close()
    print(f'Wrote scores to {args.out}')
