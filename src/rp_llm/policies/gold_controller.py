from __future__ import annotations


class GoldController:
    def __init__(self, gain_x: float = 0.2, gain_y: float = 0.2) -> None:
        self.gain_x = gain_x
        self.gain_y = gain_y

    def action(self, obs: dict) -> tuple[float, float]:
        return obs["dx"] * self.gain_x, obs["dy"] * self.gain_y


class ConservativeVariant:
    def __init__(self, gain_x: float = 0.12, gain_y: float = 0.12) -> None:
        self.gain_x = gain_x
        self.gain_y = gain_y

    def action(self, obs: dict) -> tuple[float, float]:
        return obs["dx"] * self.gain_x, obs["dy"] * self.gain_y


class LowLightVariant:
    def __init__(self, gain_x: float = 0.18, gain_y: float = 0.10) -> None:
        self.gain_x = gain_x
        self.gain_y = gain_y

    def action(self, obs: dict) -> tuple[float, float]:
        return obs["dx"] * self.gain_x, obs["dy"] * self.gain_y


class HighNoiseVariant:
    def __init__(self, gain_x: float = 0.25, gain_y: float = 0.25) -> None:
        self.gain_x = gain_x
        self.gain_y = gain_y

    def action(self, obs: dict) -> tuple[float, float]:
        return obs["dx"] * self.gain_x, obs["dy"] * self.gain_y
