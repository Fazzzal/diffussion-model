import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}

        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name] = parameter.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name].mul_(self.decay)
                self.shadow[name].add_(
                    parameter.detach(),
                    alpha=1.0 - self.decay
                )

    def state_dict(self):
        return {
            name: value.clone()
            for name, value in self.shadow.items()
        }

    def load_state_dict(self, state_dict):
        for name in self.shadow:
            if name in state_dict:
                self.shadow[name].copy_(
                    state_dict[name]
                )

    def copy_to(self, model):
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    parameter.copy_(
                        self.shadow[name]
                    )