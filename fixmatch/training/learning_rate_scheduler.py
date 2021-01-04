import tensorflow as tf
# from official.vision.image_classification.learning_rate import WarmupDecaySchedule, PiecewiseConstantDecayWithWarmup


class ConstantWarmupDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A wrapper for LearningRateSchedule that includes warmup steps."""

  def __init__(
      self,
      learning_rate: float,
      warmup_steps: int):
    """Add warmup decay to a learning rate schedule.
    Args:
      lr_schedule: base learning rate scheduler
      warmup_steps: number of warmup steps
    """
    super(ConstantWarmupDecaySchedule, self).__init__()
    self._learning_rate = learning_rate
    self._warmup_steps = warmup_steps

  def __call__(self, step: int):
    lr = self._learning_rate
    if self._warmup_steps:
      initial_learning_rate = tf.convert_to_tensor(self._learning_rate)
      dtype = initial_learning_rate.dtype
      global_step_recomp = tf.cast(step, dtype)
      warmup_steps = tf.cast(self._warmup_steps, dtype)
      warmup_lr = initial_learning_rate * global_step_recomp / warmup_steps
      lr = tf.cond(global_step_recomp < warmup_steps,
                   lambda: warmup_lr,
                   lambda: lr)
    return lr


def learning_rate_scheduler(scheduler_type, scheduler_params):
    """Build a learning rate scheduler.
    Args:
      scheduler_type: type of learning rate scheduling :
      scheduler_params: learning rate scheduler parameters.
      Examples:

          Exponential decay scheduler : EXPONENTIAL_DECAY
            {
                'initial_value': 0.004,
                'decay_steps': 1000,
                'decay_factor': 0.95
                'staircase: False
            },

          Piecewise constant scheduler : PIECEWISE_CONSTANT
            {
                'boundaries': [100000, 110000],
                'values': [1.0, 0.5, 0.1]
            },

          Constant learning rate : CONSTANT
            {
                'learning_rate': 0.004
            }

    Returns:
      Returns a learning rate scheduler
    """

    if scheduler_type == 'CONSTANT':

        return scheduler_params['learning_rate']

    elif scheduler_type == 'PIECEWISE_CONSTANT':

        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(**scheduler_params)

    elif scheduler_type == 'EXPONENTIAL_DECAY':

        return tf.keras.optimizers.schedules.ExponentialDecay(**scheduler_params)

    elif scheduler_type == 'CONSTANT_WARMUP':

        return ConstantWarmupDecaySchedule(**scheduler_params)

    elif scheduler_type == 'COSINE_DECAY':

        return tf.keras.experimental.CosineDecay(**scheduler_params)


    
    else:

        raise Exception("Invalid learning rate scheduler")

