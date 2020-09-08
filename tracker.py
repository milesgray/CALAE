"""
# MULTI-LOSS WEIGHTING WITH COEFFICIENT OF VARIATIONS
## https://arxiv.org/pdf/2009.01717.pdf

In other words, this hypothesis says that a loss with a constant value should not be optimised any further. Variance alone,
however, is not sufficient, given that it can be expected that a loss which has a larger (mean) magnitude, also has a higher
absolute variance. Even if the loss is relatively less variant. Therefore, we propose to use a dimensionless measure of
uncertainty, the Coefficient of Variation (cv, which shows the variability of the data in relation to the (observed) mean:
```
cv = σ/µ, (2)
```
where `µ` denotes the mean and `σ` the standard deviation. It allows to fairly compare the uncertainty in losses, even
with different magnitudes, under the assumption that each loss measures in a ratio-scale, that is with a unique and
non-arbitrary zero value.

Here a more robust loss ratio is proposed:
```
          Li(t)
li(t) = --------
        µLi(t − 1) 
```
*(3)*
where `µLi(t − 1)` is the mean over all observed losses from iteration 1 to (`t` - 1) for a specific loss Li. The loss ratio l(t)
has the same meaningful zero point when `Li(t)` is zero and is a relative comparison of two measurements of the loss
statistic. Now, the loss ratio is used as a point estimate of the mean to yield the following definition for loss weights:
```
     σli(t)
αi = ------
      li(t)
```
*(4)*
where `σli(t)` is the standard deviation over all known loss ratios `Li(t)/µli(t−1)` until iteration `t` - 1
"""
import numpy as np
import torch

#epoch_len = steps_per_scale[valid_scales[0]]
class LossTracker:
    def __init__(self, name, experiment, weight=1, warmup=np.inf, max=np.inf, block_size=100):
        self.name = name
        self.exp = experiment
        self.weight = weight
        self.max = max
        self.warmup = warmup
        self.block_size = block_size
        self.reset()

    def reset(self):
        self.mean = 1
        self.var = 0
        self.std = 0
        self.ratio = 0
        self.ratio_std = 0
        self.cov = 0
        self.cov_weight = self.weight
        self.value_history = np.empty(self.block_size)
        self.ratio_history = np.empty(self.block_size)
        self.max_history_size = self.block_size
        self.value = 0
        self.total = 0
        self.count = 0

    def expand_buffer(self, block_size=None):
        if block_size is not None:
            self.block_size = block_size

        self.value_history = np.concat(self.value_history, np.empty(self.block_size))
        self.ratio_history = np.concat(self.ratio_history, np.empty(self.block_size))
        self.max_history_size += self.block_size        
    
    def update(self, value, do_backwards=True, do_comet=True, do_console=False):
        if do_backwards:
            value = self.constrain_loss(value)
            value.backward()
            self.value = value.item()
        else:
            self.value = value
        self.total += self.value   
        assert self.count < self.max_history_size
        self.value_history[self.count] = self.value
        # calculate li(t)
        if self.mean != 0:             
            self.ratio = self.value / self.mean  # µLi(t − 1) is the mean over all observed losses from iteration 1 to (t - 1) for a specific loss Li
        else:
            self.ratio = 1 # ratio of 1 when mean is 0
        self.ratio_history[self.count] = self.ratio 
        self.count += 1
        if self.count > 1:  # only once there is a history          
            self.ratio_std = self.ratio_history.std() # σli(t) is the standard deviation over all known loss ratios Li(t)/µli(t−1) until iteration t - 1        
            self.cov_weight = self.ratio_std / self.ratio # αi = σli(t) / li(t)
        if self.count > self.warmup:
            # use cov weight as functioning weight after warmup period to allow for meaningful statistics to build
            self.weight = self.cov_weight        
        self.mean = self.value_history[:self.count].mean()
        self.var = self.value_history[:self.count].var()
        self.std = self.value_history[:self.count].std()
        self.cov = self.std / self.mean
        # update comet or print out
        self.log(comet=do_comet, console=do_console)

    def log(self, comet=True, console=False):        
        if comet:
            self.exp.log_metric(f"{self.name}_loss", self.value)
            self.exp.log_metric(f"{self.name}_cov", self.cov)
            self.exp.log_metric(f"{self.name}_cov_weight", self.cov_weight)
            self.exp.log_metric(f"{self.name}_var", self.var)
            self.exp.log_metric(f"{self.name}_std", self.std)
        if console:
            msg = f"[{self.name}] [{self.count}]\t{self.value} @ {self.cov_weight}x \t ~ mean: {self.mean} var: {self.var} std: {self.std} cov: {self.cov}"
            print(msg)
            self.exp.log_text(msg)

    def get_history(self):
        return self.value_history[:self.count]

    def constrain_loss(self, loss):
        loss *= self.weight
        if loss > self.max:
            magnitude = torch.floor(loss / self.max)
            loss = loss / min(magnitude, 1)
        loss = torch.clamp(loss, 0, self.max)
        return loss