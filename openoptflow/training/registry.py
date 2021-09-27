from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD, Adadelta, Adagrad, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)

from ..utils import Registry

loss_functions = Registry("loss_functions")
optimizers = Registry("optimizers")
schedulers = Registry("schedulers")

loss_functions.register(CrossEntropyLoss, "CrossEntropyLoss")
loss_functions.register(MSELoss, "MSELoss")

optimizers.register(SGD, "SGD")
optimizers.register(Adam, "Adam")
optimizers.register(AdamW, "AdamW")
optimizers.register(Adagrad, "Adagrad")
optimizers.register(Adadelta, "Adadelta")
optimizers.register(RMSprop, "RMSprop")

schedulers.register(CosineAnnealingLR, "CosineAnnealingLR")
schedulers.register(CosineAnnealingWarmRestarts, "CosineAnnealingWarmRestarts")
schedulers.register(CyclicLR, "CyclicLR")
schedulers.register(MultiStepLR, "MultiStepLR")
schedulers.register(ReduceLROnPlateau, "ReduceLROnPlateau")
schedulers.register(StepLR, "StepLR")
