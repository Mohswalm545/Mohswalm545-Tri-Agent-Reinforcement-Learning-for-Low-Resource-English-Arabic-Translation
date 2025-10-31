# 🧠 STJ — Tri-Agent Reinforcement Learning for Low-Resource English→Arabic Translation

This repository introduces **STJ**, a *Student–Teacher–Judge* (tri-agent) reinforcement learning framework that enables a model to **learn Arabic meaning, grammar, and fluency** through **self-play** — without large supervised datasets.

---

## 🔍 Overview

**Core Idea:**  
Three interacting agents form a closed training loop:

| Agent | Role | Goal |
|--------|------|------|
| **A (Student)** | Generates Arabic words or sentences. | Maximize correctness reward from the Judge. |
| **B (Teacher / Adversary)** | Provides increasingly difficult or misleading English examples. | Minimize Student’s success rate. |
| **C (Judge / Grounded Evaluator)** | Scores each translation based on accuracy, grammar, and fluency. | Provide stable, fair rewards (+1 / 0 / –1). |

<p align="center">
  <img src="https://github.com/Mohswalm545/Mohswalm545-Tri-Agent-Reinforcement-Learning-for-Low-Resource-English-Arabic-Translation/blob/main/projectimage.png" width="650">
  <br>
  <em>Figure: STJ Tri-Agent Reinforcement Learning System</em>
</p>


Training phases:
1. **Supervised Warm-Up:** A learns basic English→Arabic mappings from a bilingual lexicon.  
2. **Reinforcement Phase:** A and B self-play while C evaluates correctness.  
3. **Curriculum Growth:** Tasks scale from single words → phrases → sentences.

---

## ⚙️ Code Structure

| File | Description |
|------|--------------|
| `teacher.py` | Adversarial teacher (B) — generates English words, adjusts difficulty, adds fakes. |
| `student.py` | Simple baseline student (A). |
| `student_rl.py` | RL-based student (A) — GRU policy trained with REINFORCE + imitation warm-up. |
| `judge_translation.py` | Rule-based judge (C) for one-word translation. |
| `judge.py` | Transformer-based judge (C) with multi-component rubric scoring. |
| `env_rl.py` | RL environment connecting A, B, and C. |
| `loop.py` | Main tri-agent training loop with logging. |

---

## 🧠 Conceptual Inspiration

The **STJ** architecture combines:
- *Knowledge distillation* (Hinton et al., 2015)
- *Sequence-level RL* (Ranzato et al., 2016)
- *Adversarial training* (Goodfellow et al., 2014)
- *AI safety via debate* (Irving & Christiano, 2018)
- *Curriculum learning* (Bengio et al., 2009)
- *Self-play reinforcement learning* (Silver et al., 2016, 2018)

---

## 📊 Planned Extensions
- Multi-judge ensemble for fairness and stability.  
- Teacher specialization by linguistic phenomenon (gender, case, negation).  
- Counterfactual diacritic perturbations.  
- Arabic↔English translation self-play (“AlphaZero for translation”).  

---

## 📜 License

This work is licensed under the  
**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).**  
You may not use this material for commercial purposes.  
[Full license text](https://creativecommons.org/licenses/by-nc-sa/4.0/)

Copyright © 2025 Mohammad A. Sweilmeen

---

## 📚 References

Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531  

Ranzato, M. et al. (2016). *Sequence Level Training with Recurrent Neural Networks.* arXiv:1511.06732  

Bahdanau, D., et al. (2017). *Actor-Critic for Sequence Prediction.* arXiv:1607.07086  

Goodfellow, I., et al. (2014). *Generative Adversarial Nets.* arXiv:1406.2661  

Irving, G., Christiano, P., et al. (2018). *AI Safety via Debate.* arXiv:1805.00899  

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). *Curriculum Learning.* *ICML 2009.*  

Silver, D., et al. (2016). *Mastering the Game of Go with Deep Neural Networks and Tree Search.* *Nature*, 529(7587), 484–489.  

Silver, D., et al. (2018). *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play.* *Science*, 362(6419), 1140–1144.  

Rei, R., et al. (2022). *COMET-22: A Reference-Free Evaluation Metric for Machine Translation.* arXiv:2209.15649  

Sweilmeen, M. A. (2025). *Tri-Agent Reinforcement Learning for Low-Resource English→Arabic Translation (GitHub Repository).*  
[https://github.com/Mohswalm545/Mohswalm545-Tri-Agent-Reinforcement-Learning-for-Low-Resource-English-Arabic-Translation](https://github.com/Mohswalm545/Mohswalm545-Tri-Agent-Reinforcement-Learning-for-Low-Resource-English-Arabic-Translation)

