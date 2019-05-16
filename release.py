from utils import point2knob, knob2point
from sampling import clustering, get_samples

class ReLeASE(Tuner):
    def __init__(self, task, cost_model, agent, plan_size): 
        
        # space
        self.task = task
        self.target = task.target
        self.plan_size = plan_size
        self.space = task.config_space
        self.space_len = len(task.config_space)
        self.dims = [len(x) for x in self.space.space_map.values()]

        self.cost_model = cost_model
        self.model_optimizer = model_optimizer
        self.diversity_filter_ratio = diversity_filter_ratio

        if self.diversity_filter_ratio:
            assert self.diversity_filter_ratio >= 1, "Diversity filter ratio " \
                                                     "must be larger than one"
        self.adaptive = True
        self.all = plan_size

        # trial plan
        self.trials = []
        self.trial_pt = 0
        self.visited = set()

        # observed samples
        self.xs = []
        self.ys = []
        self.flops_max = 0.0
        self.train_ct = 0

    def next_batch(self, batch_size, adaptive=False):
        ret = []

        if adaptive == True:
            self.diversity_filter_ratio = 2
            self.adaptive = True

        counter = 0
        while counter < batch_size:
            if len(self.visited) >= len(self.space):
                break

            while self.trial_pt < len(self.trials):
                index = self.trials[self.trial_pt]
                if index not in self.visited:
                    break
                self.trial_pt += 1

            if self.trial_pt >= len(self.trials) - int(0.05 * self.plan_size):
                index = np.random.randint(len(self.space))
                while index in self.visited:
                    index = np.random.randint(len(self.space))

            ret.append(self.space.get(index))
            self.visited.add(index)

            counter += 1
            #print(ret)
        return ret

    def update(self, inputs, results):
        updated = 0

        for inp, res in zip(inputs, results):
            index = inp.config.index
            if res.error_no == 0:
                self.xs.append(index)
                flops = inp.task.flop / np.mean(res.costs)
                self.flops_max = max(self.flops_max, flops)
                self.ys.append(flops)
            else:
                self.xs.append(index)
                self.ys.append(0.0)

        if len(self.xs) >= self.all \
                and self.flops_max > 1e-6:

            self.cost_model.fit(self.xs, self.ys, self.plan_size)

            start_time = time.time()
            if self.diversity_filter_ratio:
                candidate = self.model_optimizer.find_maximums(
                    self.cost_model, self.plan_size * self.diversity_filter_ratio, self.visited)
                scores = self.cost_model.predict(candidate)
                knobs = [point2knob(x, self.dims) for x in candidate]
                pick_index = submodular_pick(0 * scores, knobs, self.plan_size, knob_weight=1)
                maximums = np.array(candidate)[pick_index]
            else:
                maximums = self.model_optimizer.find_maximums(
                    self.cost_model, self.plan_size, self.visited)

            if self.adaptive == True:
                no_d = 3
                dims = self.dims
                good_dims = np.argsort(-np.array(dims))[:no_d]

                points = [point2knob(config, self.dims) for config in maximums]
                samples = [tuple(np.array(point)[good_dims]) for point in points]

                last_loss = 99999999
                for k in range(8, 65, 8):
                    centroids, cluster, loss = clustering(samples, k)

                    # to determine knee for the trade-off
                    if loss >= last_loss / 2.5:
                        break
                    else:
                        last_loss = loss
                updated = 1

                reduced_samples = get_samples(points, dims, good_dims, centroids, cluster)
                maximums = [knob2point(sample, dims) for sample in reduced_samples]

            self.trials = maximums
            self.trial_pt = 0
            self.train_ct += 1

            if self.adaptive == True:
                self.all += len(self.trials)

        return updated
