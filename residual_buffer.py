
from typing import Literal

import numpy as np
import openai

import utils


client = ...  # TODO

class ResidualBuffer:
    def __init__(self, intent: str, buf_size: int = 1, model: str = "gpt-3.5-turbo-instruct") -> None:
        self.intent = intent
        self.buf_size = buf_size
        self.model = model
        self._attacks: list[str] = []
        self._scores: list[float] = []
        self._responses: list[str] = []

    def push(self, attack: str, response: str) -> None:
        if len(self._attacks) < self.buf_size:
            self._attacks.append(attack)
            self._responses.append(response)
            self._scores.append(utils.judge_prompt(client, self.intent, response, self.model))
        else:
            min_score = min(self._scores)
            min_score_idx = self._scores.index(min_score)
            score = utils.judge_prompt(client, self.intent, response, self.model)
            if score < min_score:
                self._attacks[min_score_idx] = attack
                self._responses[min_score_idx] = response
                self._scores[min_score_idx] = score 

    def sample(self, mode: Literal["max", "random"] = "random") -> tuple[str, str]:
        if not self._attacks:
            return ""
        
        if mode == "max":
            max_score = max(self._scores)
            max_score_idx = self._scores.index(max_score)
            return self._attacks[max_score_idx], self._responses[max_score_idx]
        elif mode == "random":
            idx = np.random.choice(np.arange(len(self._scores)))
            return self._attacks[idx], self._responses[idx]
