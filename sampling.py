# p0 and p1 are tuples
def distance(p0, p1):
    return np.sum((np.array(p0) - np.array(p1))**2).astype(float)

def clustering(points, k, max_iter=100):
    # initialize centroids and clusters
    centroids = [points[i] for i in np.random.randint(len(points), size=k)]

    cluster = [0] * len(points)
    prev_cluster = [-1] * len(points)

    # start
    i = 0
    force_recalculation = False
    while (cluster != prev_cluster) or (i > max_iter) or (force_recalculation):
        if i > max_iter * 2:
            break

        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1

        # update cluster
        for p in range(0, len(points)):
            min_dist = float("inf")

            # check for min distance to assign cluster
            for c in range(0, len(centroids)):
                dist = distance(points[p], centroids[c])

                if dist < min_dist:
                    min_dist = dist
                    cluster[p] = c

        # update centroids
        for c in range(0, len(centroids)):
            new_centroid = [0] * len(points[0])
            members = 0

            # add
            for p in range(0, len(points)):
                if cluster[p] == c:
                    for j in range(0, len(points[0])):
                        new_centroid[j] += points[p][j]
                    members += 1

            # divide
            for j in range(0, len(points[0])):
                if members != 0:
                    new_centroid[j] = int(new_centroid[j] / float(members))

                # force recalculation
                else:
                    new_centroid = random.choice(points)
                    force_recalculation = True

            centroids[c] = tuple(new_centroid)

    loss = 0
    for p in range(0, len(points)):
        loss += distance(points[p], centroids[cluster[p]])

    return centroids, cluster, loss

# two strategy possible
# (1) mode of each dimension
# (2) NN
def get_samples(points, dims, good_dims, centroids, cluster):
    samples = []
    for c in range(0, len(centroids)):
        #print centroids[c]

        # find members
        members = []
        for p in range(0, len(points)):
            if cluster[p] == c:
                members.append(points[p])
        #print members

        # calculate other coordinates
        sample_dims = {}
        if len(members) == 0:
            for d in range(0, len(points[0])):
                sample_dims[d] = range(0, dims[d])
        else:
            for d in range(0, len(points[0])):
                if d in good_dims:
                    continue

                if d not in sample_dims:
                    sample_dims[d] = []

                for m in members:
                    sample_dims[d].append(m[d])

        # construct sample
        sample = []
        for d in range(0, len(points[0])):
            if d in good_dims:
                sample.append(centroids[c][list(good_dims).index(d)])
                continue

            # deterministic
            #sample.append(max(set(sample_dims[d]), key=sample_dims[d].count))

            # sampling
            sample.append(random.sample(sample_dims[d], 1)[0])

        done = False
        while done is False:
            if sample not in samples:
                samples.append(sample)
                done = True
            else:
                sample = random.sample(points, 1)[0]

        #print sample_dims
        #for d in sample_dims:
        #    print max(set(sample_dims[d]), key=sample_dims[d].count)

        #print sample
    #print len(samples)

    return samples
