from collections import defaultdict, Counter

import math


def adamicadar(followers, users, i, j):
    if i == j:
        return 0
    listaA = followers.get(i)
    listaB = followers.get(j)
    if listaA is None or listaB is None:
        return 0
    intersect = [x for x in listaA if x in listaB]
    result = 0
    for user in intersect:
        svcs = users.get(user)
        result += 1.0 / math.log(len(svcs))
    return result


def jaccard(followers, users, i, j):
    if i == j:
        return 1
    listaA = followers.get(i)
    listaB = followers.get(j)
    if listaA is None or listaB is None:
        return 0
    intersect = 0
    union = len(listaB)
    for user in listaA:
        intersect += 1 if user in listaB else 0
        union += 1 if user not in listaB else 0
    return intersect / union


def commonneighbours(followers, users, i, j):
    if i == j:
        return 1
    listaA = followers.get(i)
    listaB = followers.get(j)
    if listaA is None or listaB is None:
        return 0
    intersect = 0
    for user in listaA:
        intersect += 1 if user in listaB else 0
    return intersect


def hpi(followers, users, i, j):
    if i == j:
        return 1
    listaA = followers.get(i)
    listaB = followers.get(j)
    if listaA is None or listaB is None:
        return 0
    intersect = 0
    for user in listaA:
        intersect += 1 if user in listaB else 0
    return intersect / (len(listaA) if len(listaA) < len(listaB) else len(listaB))


def hdi(followers, users, i, j):
    if i == j:
        return 1
    listaA = followers.get(i)
    listaB = followers.get(j)
    if listaA is None or listaB is None:
        return 0
    intersect = 0
    for user in listaA:
        intersect += 1 if user in listaB else 0
    return intersect / (len(listaA) if len(listaA) > len(listaB) else len(listaB))


def salton(followers, users, i, j):
    if i == j:
        return 1
    listaA = followers.get(i)
    listaB = followers.get(j)
    if listaA is None or listaB is None:
        return 0
    intersect = 0
    for user in listaA:
        intersect += 1 if user in listaB else 0
    return intersect / math.sqrt(len(listaA) * len(listaB))


def lhni(followers, users, i, j):
    if i == j:
        return 1
    listaA = followers.get(i)
    listaB = followers.get(j)
    if listaA is None or listaB is None:
        return 0
    intersect = 0
    for user in listaA:
        intersect += 1 if user in listaB else 0
    return intersect / (len(listaA) * len(listaB))


def pa(followers, users, i, j):
    if i == j:
        return 1
    listaA = followers.get(i)
    listaB = followers.get(j)
    if listaA is None or listaB is None:
        return 0
    return len(listaA) * len(listaB)


def sorensen(followers, users, i, j):
    if i == j:
        return 1
    listaA = followers.get(i)
    listaB = followers.get(j)
    if listaA is None or listaB is None:
        return 0
    intersect = 0
    for user in listaA:
        intersect += 1 if user in listaB else 0
    return 2 * intersect / (len(listaA) * len(listaB))
