#@title LossTracker
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
import torch
import numpy as np

class LossTracker:
    def __init__(self, name, experiment, weight=1., 
                 warmup=np.inf, stats_window=100,
                 max_loss=np.inf, min_loss=0., block_size=100, 
                 use_cov_weight=False, use_scaling=False, scale_range=[0.2, 5],
                 use_magnitude_after=np.inf):
        """A container that tracks various metrics related to a running stream of loss values during training of a PyTorch network.

        A constraint is applied to the loss to ensure the value is positive by default,
        but this behavior is controlled by the min_loss and max_loss arguments. The `use_magnitude`
        argument will apply a scaling factor to the loss based on how much larger the actual value
        is compared to the max value.

        Args:
            name (string): The name used for display purposes, should be unique amongst all losses declared.
            experiment (comet_ml.Experiment): The experiment object used to transmit each value to comet.
            weight (float, optional): Loss value is always multiplied by this amount. Defaults to 1.
            loss_limit (list, optional): The minimum and maximum values to restrict
                final loss values to. Defaults to [-np.inf, np.inf].
            warmup (int, optional): Start to apply advanced scaling after this many values have been
                recorded, the default effectively disables the advanced scaling features. Defaults to np.inf.
            stats_window (int, optional): Number of historic values to use when calculating
                statistics such as mean in a moving style. 
                Use `None` to allow for all historic values to be used. Defaults to 100.
            max_loss (float, optional): The largest value allowed for the loss. When `use_magnitude`
                is enabled, the final value used can be above this. Defaults to np.inf.
            min_loss (float, optional): The smallest value allowed for the loss. Defaults to 0.
            block_size (int, optional): Size of the buffer array allocated, larger values will use
                more memory but may be more performant otherwise. Defaults to 100.
            use_cov_weight (bool, optional): Use the covarience weighting feature that dynamically
                changes the weight value based on the running statistics. Defaults to False.
            use_scaling (bool, optional): 
                Scale the loss based on a running ratio value. Defaults to False.
            scale_range (list, optional): Minimum and maximum values to restrict dynamic scaling by. 
                This prevents massive scaling values for irregular loss functions. 
                Defaults to [0.2, 5].
            use_magnitude_after (int, optional): Scale a max loss by how much bigger than the max the original
                value was after this many consecutive values that are constrained by the max. Defaults to False.
        """
        self.name = name
        self.exp = experiment
        self.weight = weight
        self.max = max_loss
        self.min = min_loss
        self.warmup = warmup
        self.stats_window = stats_window
        self.block_size = block_size
        self.use_scaling = use_scaling
        self.scale_min = scale_range[0]
        self.scale_max = scale_range[1]
        self.use_cov_weight = use_cov_weight        
        self.use_magnitude_after = use_magnitude_after
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
        self.max_seen_raw = 0
        self.max_seen_constrained = 0
        self.consecutive_max_seen = 0
        self.is_value_max = False

    def expand_buffer(self, block_size=None):
        if block_size is not None:
            self.block_size = block_size

        empty = np.empty(self.block_size)

        if not isinstance(self.value_history, np.ndarray):
            self.value_history = np.array(self.value_history)
        if not isinstance(self.ratio_history, np.ndarray):
            self.ratio_history = np.array(self.ratio_history)
        try:
            self.value_history = np.concatenate([self.value_history, empty.copy()])
        except Exception as e:
            print(f"failed to expand value history ({type(self.value_history)})") 
            temp = np.empty(self.value_history.shape[0] + self.block_size)
            temp[:self.value_history.shape[0]] = self.value_history
            self.value_history = temp
            del temp

        try:
            self.ratio_history = np.concatenate([self.ratio_history, empty.copy()])
        except Exception as e:
            print(f"failed to expand ratio history ({type(self.ratio_history)})")
            temp = np.empty(self.ratio_history.shape[0] + self.block_size)
            temp[:self.ratio_history.shape[0]] = self.ratio_history
            self.ratio_history = temp
            del temp
        self.max_history_size += self.block_size        
    
    def update(self, value, do_backwards=True, do_comet=True, do_console=False):
        if self.use_scaling and self.count > self.warmup:
            # Applies dynamic scaling based on running statistics
            value = self.scale_loss(value)
        # Applies static weighting to value
        value = self.adjust_loss(value)
        # Applies (self.min, self.max) constraint to avoid explosion of value and negative values (as default)
        value = self.constrain_loss(value)
        if do_backwards:
            # Apply backpropagation on the pytorch Tensor version of the loss
            value.backward()
        # Get scalar value from pytorch Tensor
        self.value = value.item()

        self.total += self.value
        if self.count == self.max_history_size: 
            # Buffer is full, allocate a new block of numpy array memory
            self.expand_buffer()
        assert self.count < self.max_history_size
        self.value_history[self.count] = self.value

        self.set_stats()
        
        # calculate li(t)
        if self.mean != 0:             
            self.ratio = self.value / self.mean  # µLi(t − 1) is the mean over all observed losses from iteration 1 to (t - 1) for a specific loss Li
        else:
            self.ratio = 1 # ratio of 1 when mean is 0
        self.ratio = min(max(self.ratio, self.scale_min), self.scale_max)
        self.ratio_history[self.count] = self.ratio 
        self.count += 1
        if self.count > 1:  # only once there is a history          
            self.ratio_std = self.ratio_history[:self.count].std() # σli(t) is the standard deviation over all known loss ratios Li(t)/µli(t−1) until iteration t - 1        
            self.cov_weight = self.ratio_std / self.ratio # αi = σli(t) / li(t)
        if self.use_cov_weight and self.count > self.warmup:
            # use cov weight as functioning weight after warmup period to allow for meaningful statistics to build
            self.weight = self.cov_weight     

        # update comet or print out
        self.log(comet=do_comet, console=do_console)

    def set_stats(self):
        if self.count == 0: 
            return
        try:
            start = self.count - self.stats_window if self.stats_window else 0
            start = start if start > 0 else 0
            end = self.count
            self.max = self.value_history[start:end].max()
            self.min = self.value_history[start:end].min()
            self.mean = self.value_history[start:end].mean()
            self.var = self.value_history[start:end].var()
            self.std = self.value_history[start:end].std()
            self.cov = self.std / self.mean
        except Exception as e:
            print(f"[ERROR]\tFailed to set metric stats\n{e}")        

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

    def scale_loss(self, loss):
        loss *= self.ratio
        return loss

    def adjust_loss(self, loss):
        loss *= self.weight
        return loss

    def constrain_loss(self, loss):        
        if loss > self.max_seen_raw:
            # Keeps track of how big the loss is actually getting before constraining it
            self.max_seen_raw = loss
        # Track how long this loss is stuck at the max value
        if self.is_value_max:
            self.consecutive_max_seen += 1
        else:
            self.consecutive_max_seen = 0

        if loss > self.max:
            # Multiplier of how much bigger loss is than the max
            magnitude = loss / self.max
            self.is_value_max = True
        else:
            magnitude = 1.0            
            self.is_value_max = False
        
        loss = torch.clamp(loss, 0, self.max)
        if self.use_magnitude_after > self.consecutive_max_seen:
            loss = loss * max(magnitude, 1)
        if loss > self.max_seen_constrained:
            # Keeps track of how big the loss gets after being constrained
            self.max_seen_constrained = loss
        return loss