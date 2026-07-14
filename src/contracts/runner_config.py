from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """Configuration for execution runners."""
    
    checkpoint_interval: int
    max_consecutive_failures: int
