import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warmup and restarts learning rate scheduler.
    We restart the learning rate schedule after a certain number of steps,
    increasing the cycle length and decreasing the maximum learning rate.
    This way the optimizer explores aggressively, converges, and then explores
    again from a new angle, escaping local minima.
    We increase the cycle length because in early stages we want to explore 
    aggressively and have frequent restarts, while in later stages we want
    longer cycles, more refinement, and fewer disruptions.

    Input:
        optimizer: torch optimizer
        first_cycle_steps: int - number of steps for the first cycle
        cycle_mult: float - factor to increase cycle steps after each cycle
        max_lr: float - maximum learning rate
        min_lr: float - minimum learning rate
        warmup_steps: int - number of warmup steps
        gamma: float - factor to decrease max_lr after each cycle
        last_epoch: int - last epoch index
    """

    def __init__(self, optimizer: torch.optim.Optimizer, first_cycle_steps: int, cycle_mult: float = 1.0,
                 max_lr: float = 0.1, min_lr: float = 0.001, warmup_steps: int = 0,
                 gamma: float = 1.0, last_epoch: int = -1):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step count in the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate to min_lr at the beginning
        self.init_lr()

    def init_lr(self):
        """
        Initialize learning rates to min_lr, respecting per-group LR ratios.

        Each param group may have a different initial LR (e.g. differential LR
        for newly added modules).  We compute a per-group ratio so the cosine
        schedule scales each group proportionally:
            ratio = initial_lr / global_max_lr
            pg_min_lr = global_min_lr * ratio
            pg_max_lr = global_max_lr * ratio  (applied in get_lr via self.max_lr * ratio)
        """
        self.base_lrs = []  # per-group min learning rates
        self.lr_ratios = []  # per-group scaling factors
        for param_group in self.optimizer.param_groups:
            # initial_lr is saved by _LRScheduler.__init__ from the optimizer
            initial_lr = param_group.get('initial_lr', self.max_lr)
            ratio = initial_lr / self.max_lr if self.max_lr > 0 else 1.0
            pg_min_lr = self.min_lr * ratio
            param_group['lr'] = pg_min_lr
            self.base_lrs.append(pg_min_lr)
            self.lr_ratios.append(ratio)

    def get_lr(self):
        """
        Compute learning rates for the current step.
        Each param group is scaled by its own ratio so differential LRs are preserved.
        """
        if self.step_in_cycle == -1:  # before the first step
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:  # in warmup phase
            return [
                (self.max_lr * ratio - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr, ratio in zip(self.base_lrs, self.lr_ratios)
            ]
        else:  # in cosine annealing phase
            return [
                base_lr + (self.max_lr * ratio - base_lr) * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps))) / 2
                for base_lr, ratio in zip(self.base_lrs, self.lr_ratios)
            ]

    def step(self, epoch=None):
        """
        Update learning rates and cycle information.
        """
        if epoch is None: # if epoch is not provided, step by 1
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1

            if self.step_in_cycle >= self.cur_cycle_steps: # end of current cycle
                # restart the cycle
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps

        else: 
            if epoch >= self.first_cycle_steps: 
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * (self.cycle_mult ** n)
            else:
                # still in the first cycle, no restarts yet
                self.cycle = 0
                self.step_in_cycle = epoch
                self.cur_cycle_steps = self.first_cycle_steps
                
        
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr