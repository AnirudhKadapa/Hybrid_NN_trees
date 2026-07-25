from pathlib import Path

class GlobalBest:
    def __init__(self):
        self.val_acc = float('-inf')
        self.state = None
        self.config = None
        self.trial_number = None
        self.best_epoch = None

    def update(self, *, val_acc, state, config, trial_number, best_epoch): 
        if val_acc <= self.val_acc:
            return False
        self.val_acc = val_acc
        self.config = config
        self.trial_number = int(trial_number)
        self.best_epoch = int(best_epoch)

        self.state = {name:value.cpu().clone() for name, value in state.items()}

        return True
