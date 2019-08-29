from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()
writer.add_scalar('Loss/train', .1, 0)
writer.add_scalar('Loss/train', .5, 1)
writer.add_scalar('Loss/train', .10, 10)
writer.flush()