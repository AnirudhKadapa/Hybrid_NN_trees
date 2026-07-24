from pathlib import Path

class GlobalBest:
    def __init__(self, path:Path):
        self.val_acc = float('-inf')
        self.state = None
        self.config = None
        self.trail_number = None
        self.best_epoch = None

    def update(self, *, val_acc, state, config, trail_number, best_epoch): 
        if val_acc <= self.val_acc:
            return False
        self.val_acc = val_acc
        self.config = config
        self.trail_number = int(trail_number)
        self.best_epoch = int(best_epoch)

        self.state = {name:value.cpu().clone() for name, value in state.items()}

        return True
