from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
from torch.optim import SGD, Adadelta, Adagrad, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

from ..functional import CosineWarmupScheduler
from ..utils import Registry

loss_functions = Registry("loss_functions")
optimizers = Registry("optimizers")
schedulers = Registry("schedulers")

loss_functions.register(CrossEntropyLoss, "CrossEntropyLoss")
loss_functions.register(MSELoss, "MSELoss")
loss_functions.register(L1Loss, "L1Loss")

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
schedulers.register(OneCycleLR, "OneCycleLR")
schedulers.register(CosineWarmupScheduler, "CosineWarmupScheduler")
