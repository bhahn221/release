import numpy as np

from utils import knob2point, point2knob
from measure import run_hardware, begin_tuning, end_tuning

class Tuner(object):
    def __init__(self, task, **kwargs):
        self.task = task

        # current best
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None
        self.best_iter = 0

        # end conditions
        self.n_trial = None

        # adaptive sampling option
        self.adaptive = False
        self.adaptive_runs = 16

    def next_batch(self, batch_size):
        return

    def update(self, inputs, results):
        return

    def tune(self, n_trial):
        # set in_tuning
        begin_tuning()

        # start conditions
        self.n_trial = n_trial
        i = error_ct = 0

        if self.adaptive == True:
            # with adaptive sampling
            j = 0
            while j < self.adaptive_runs and i < n_trial:
                if not self.has_next():
                    break

                configs = self.next_batch(min(n_parallel, n_trial - i), True)
                inputs = [get_inputs(self.task.target, self.task, config) for config in configs]
                results = run_hardware(inputs)

                # keep best config
                for k, (inp, res) in enumerate(zip(inputs, results)):
                    config = inp.config
                    if res.error_no == 0:
                        flops = inp.task.flop / np.mean(res.costs)
                        error_ct = 0
                    else:
                        flops = 0
                        error_ct += 1

                    if flops > self.best_flops:
                        self.best_flops = flops
                        self.best_config = config
                        self.best_measure_pair = (inp, res)
                        self.best_iter = i + k

                i += len(results)
                j += self.update(inputs, results)

        else:
            # without adaptive sampling
            while i < n_trial:
                if not self.has_next():
                    break

                configs = self.next_batch(min(n_parallel, n_trial - i))
                inputs = [get_inputs(self.task.target, self.task, config) for config in configs]
                results = run_hardware(inputs)

                # keep best config
                for k, (inp, res) in enumerate(zip(inputs, results)):
                    config = inp.config
                    if res.error_no == 0:
                        flops = inp.task.flop / np.mean(res.costs)
                        error_ct = 0
                    else:
                        flops = 0
                        error_ct += 1

                    if flops > self.best_flops:
                        self.best_flops = flops
                        self.best_config = config
                        self.best_measure_pair = (inp, res)
                        self.best_iter = i + k

                i += len(results)
