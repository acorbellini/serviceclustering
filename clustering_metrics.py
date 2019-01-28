from collections import defaultdict, Counter

import math

def cohesiveness(sim, clusters_indices):
    clusters = defaultdict(list)
    for index, classif in enumerate(clusters_indices):
        clusters[classif].append(index)
    clustersize = Counter(clusters_indices)
    cohesiveness = 0;
    for k, cluster in clusters.items():
        clusterCohesivenes = 0;
        for svcA in cluster:
            for svcB in cluster:
                clusterCohesivenes += sim[svcA][svcB];
        clusterCohesivenes /= clustersize[k] * clustersize[k];
        cohesiveness += clusterCohesivenes * clustersize[k] / len(clusters_indices)
    return cohesiveness


def entropy(labels, clusters_indices):
    clusters = defaultdict(list)
    for index, classif in enumerate(clusters_indices):
        clusters[classif].append(index)
    clustersize = Counter(clusters_indices)
    ret = 0
    for cluster in clusters:
        countcat = Counter([labels[svc] for svc in clusters[cluster]])
        ret += clustersize[cluster] / len(clusters_indices) * sum(
            map(lambda x: -(countcat[x] / clustersize[cluster]) * math.log2(countcat[x] / clustersize[cluster]),
                countcat));
    return ret


def purity(labels, clusters_indices):
    clusters = defaultdict(list)
    for index, classif in enumerate(clusters_indices):
        clusters[classif].append(index)
    ret = 0
    for cluster in clusters:
        countcat = Counter([labels[svc] for svc in clusters[cluster]])
        max_key, num = countcat.most_common(1)[0]
        ret += num / len(clusters_indices)
    return ret


def fmeasure(labels, clusters_indices):
    clusters = defaultdict(list)
    for index, classif in enumerate(clusters_indices):
        clusters[classif].append(index)
    classfmeasures = {}
    label_size = Counter(labels)
    for cluster, services in clusters.items():
        countcat = Counter([labels[svc] for svc in services])
        for key, count in countcat.items():
            recall = count / label_size[key]
            precision = count / len(services)
            currentf = (2 * recall * precision) / (recall + precision)
            classfmeasure = classfmeasures.get(key)
            if (classfmeasure is None or classfmeasure < currentf):
                classfmeasures[key] = currentf

    fmeasure = 0
    for label, fm in classfmeasures.items():
        fmeasure += label_size.get(label) / len(clusters_indices) * fm
    return fmeasure
