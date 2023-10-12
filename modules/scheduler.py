from torch.optim.lr_scheduler import *
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import torch.optim as optim


class CustomWarmupStepLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.5, warmup_epochs=10, target_lr=0.001, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        super(CustomWarmupStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [self.target_lr * warmup_factor for _ in self.base_lrs]
        else:
            return [self.target_lr * self.gamma ** ((self.last_epoch - self.warmup_epochs) // self.step_size) for _ in self.base_lrs]


def get_scheduler(args, optimizer):
    if args.lr_scheduler == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr/100)
    elif args.lr_scheduler == 'ExponentialLR':
        return ExponentialLR(optimizer, gamma=0.9)
    elif args.lr_scheduler == 'StepLR':
        return StepLR(optimizer, step_size=args.num_epochs//5, gamma=0.5)
    elif args.lr_scheduler == 'MultiStepLR':
        return MultiStepLR(optimizer, milestones=[args.num_epochs//2, args.num_epochs*3//4, args.num_epochs*7//8], gamma=0.5)
    elif args.lr_scheduler == 'LambdaLR':
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)
    elif args.lr_scheduler == 'LinearWarmup':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_steps)
    elif args.lr_scheduler == 'CosineWarmup':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_steps)
    elif args.lr_scheduler == 'CustomWarmupStepLR':
        return CustomWarmupStepLR(optimizer, step_size=args.num_epochs//5, gamma=0.5, warmup_epochs=args.warmup_epochs, target_lr=args.lr)
    else:
        return None
    
