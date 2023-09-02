#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/8/27 10:02
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : HypE.py
# @Statement : Hypervolume estimation algorithm for multiobjective optimization
# @Reference : Bader J, Zitzler E. HypE: An algorithm for fast hypervolume-based many-objective optimization[J]. Evolutionary Computation, 2011, 19(1): 45-76.
import numpy as np
import matplotlib.pyplot as plt


def cal_obj(pop, nobj):
    # 0 <= x <= 1
    g = 100 * (pop.shape[1] - nobj + 1 + np.sum((pop[:, nobj - 1:] - 0.5) ** 2 - np.cos(20 * np.pi * (pop[:, nobj - 1:] - 0.5)), axis=1))
    objs = np.zeros((pop.shape[0], nobj))
    temp_pop = pop[:, : nobj - 1]
    for i in range(nobj):
        f = 0.5 * (1 + g)
        f *= np.prod(temp_pop[:, : temp_pop.shape[1] - i], axis=1)
        if i > 0:
            f *= 1 - temp_pop[:, temp_pop.shape[1] - i]
        objs[:, i] = f
    return objs


def hv(npop, points, nobj, bounds, pvec, alpha, k):
    # the recursive method to calculate the accurate HV
    h = np.zeros(npop)
    i = np.argsort(points[:, nobj - 1])
    S = points[i]
    pvec = pvec[i]
    for i in range(S.shape[0]):
        if i < S.shape[0] - 1:
            extrusion = S[i + 1, nobj - 1] - S[i, nobj - 1]
        else:
            extrusion = bounds[nobj - 1] - S[i, nobj - 1]
        if nobj == 1:
            if i >= k:
                break
            if np.all(alpha >= 0):
                h[pvec[: i + 1]] += extrusion * alpha[i]
        elif extrusion > 0:
            h += extrusion * hv(npop, S[: i + 1], nobj - 1, bounds, pvec[: i + 1], alpha, k)
    return h


def calHV(points, bounds, k, ns):
    # calculate the hypervolume-based fitness value of each solution
    (npop, nobj) = points.shape
    if nobj > 2:  # the estimated methods for three or more objectives
        alpha = np.zeros(npop)
        for i in range(k):
            alpha[i] = np.prod((k - np.arange(1, i + 1)) / (npop - np.arange(1, i + 1))) / (i + 1)
        Fmin = np.min(points, axis=0)
        S = np.random.uniform(np.tile(Fmin, (ns, 1)), np.tile(bounds, (ns, 1)))
        Pds = np.full((npop, ns), False)
        dS = np.zeros(ns, dtype=int)
        for i in range(npop):
            x = np.sum(np.tile(points[i], (ns, 1)) - S <= 0, axis=1) == nobj
            Pds[i, x] = True
            dS[x] += 1
        F = np.zeros(npop)
        for i in range(npop):
            F[i] = np.sum(alpha[dS[Pds[i]] - 1], axis=0)
        F *= np.prod(bounds - Fmin) / ns
    else:  # the accurate method for two objectives
        pvec = np.arange(npop)
        alpha = np.zeros(k)
        for i in range(k):
            j = np.arange(1, i + 1)
            alpha[i] = np.prod((k - j) / (npop - j)) / (i + 1)
        F = hv(npop, points, nobj, bounds, pvec, alpha, k)
    return F


def selection(pop, HV, k=2):
    # binary tournament selection
    (npop, nvar) = pop.shape
    nm = npop if npop % 2 == 0 else npop + 1  # mating pool size
    mating_pool = np.zeros((nm, nvar))
    for i in range(nm):
        selections = np.random.choice(npop, k, replace=False)
        ind = selections[np.argmax(HV[selections])]
        mating_pool[i] = pop[ind]
    return mating_pool


def crossover(mating_pool, lb, ub, eta_c):
    # simulated binary crossover (SBX)
    (noff, nvar) = mating_pool.shape
    nm = int(noff / 2)
    parent1 = mating_pool[:nm]
    parent2 = mating_pool[nm:]
    beta = np.zeros((nm, nvar))
    mu = np.random.random((nm, nvar))
    flag1 = mu <= 0.5
    flag2 = ~flag1
    beta[flag1] = (2 * mu[flag1]) ** (1 / (eta_c + 1))
    beta[flag2] = (2 - 2 * mu[flag2]) ** (-1 / (eta_c + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, (nm, nvar))
    beta[np.random.random((nm, nvar)) < 0.5] = 1
    beta[np.tile(np.random.random((nm, 1)) > 1, (1, nvar))] = 1
    offspring1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    offspring2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = np.concatenate((offspring1, offspring2), axis=0)
    offspring = np.min((offspring, np.tile(ub, (noff, 1))), axis=0)
    offspring = np.max((offspring, np.tile(lb, (noff, 1))), axis=0)
    return offspring


def mutation(pop, lb, ub, eta_m):
    # polynomial mutation
    (npop, dim) = pop.shape
    lb = np.tile(lb, (npop, 1))
    ub = np.tile(ub, (npop, 1))
    site = np.random.random((npop, dim)) < 1 / dim
    mu = np.random.random((npop, dim))
    delta1 = (pop - lb) / (ub - lb)
    delta2 = (ub - pop) / (ub - lb)
    temp = np.logical_and(site, mu <= 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (1 - delta1[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1)
    temp = np.logical_and(site, mu > 0.5)
    pop[temp] += (ub[temp] - lb[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (1 - delta2[temp]) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
    pop = np.min((pop, ub), axis=0)
    pop = np.max((pop, lb), axis=0)
    return pop


def nd_sort(objs):
    # fast non-domination sort
    (npop, nobj) = objs.shape
    n = np.zeros(npop, dtype=int)  # the number of individuals that dominate this individual
    s = []  # the index of individuals that dominated by this individual
    rank = np.zeros(npop, dtype=int)
    ind = 0
    pfs = {ind: []}  # Pareto fronts
    for i in range(npop):
        s.append([])
        for j in range(npop):
            if i != j:
                less = equal = more = 0
                for k in range(nobj):
                    if objs[i, k] < objs[j, k]:
                        less += 1
                    elif objs[i, k] == objs[j, k]:
                        equal += 1
                    else:
                        more += 1
                if less == 0 and equal != nobj:
                    n[i] += 1
                elif more == 0 and equal != nobj:
                    s[i].append(j)
        if n[i] == 0:
            pfs[ind].append(i)
            rank[i] = ind
    while pfs[ind]:
        pfs[ind + 1] = []
        for i in pfs[ind]:
            for j in s[i]:
                n[j] -= 1
                if n[j] == 0:
                    pfs[ind + 1].append(j)
                    rank[j] = ind + 1
        ind += 1
    pfs.pop(ind)
    return pfs, rank


def environmental_selection(pop, objs, npop, ref, ns):
    # HypE environmental selection
    pfs, rank = nd_sort(objs)
    selected = np.full(pop.shape[0], False)
    ind = 0
    while np.sum(selected) + len(pfs[ind]) <= npop:
        selected[pfs[ind]] = True
        ind += 1
    K = npop - np.sum(selected)
    last = np.array(pfs[ind])

    # select the remaining K solutions
    choose = np.full(len(last), True)
    while np.sum(choose) > K:
        remain = np.where(choose)[0]
        HV = calHV(objs[last[remain]], ref, np.sum(choose) - K, ns)
        ind = np.argmin(HV)
        choose[remain[ind]] = False
    selected[last[choose]] = True
    return pop[selected], objs[selected]


def dominates(obj1, obj2):
    # determine whether obj1 dominates obj2
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] > obj2[i]:
            return False
        elif obj1[i] != obj2[i]:
            sum_less += 1
    return sum_less > 0


def main(npop, iter, lb, ub, nobj=3, eta_c=20, eta_m=20, ns=10000):
    """
    The main loop
    :param npop: population size
    :param iter: iteration number
    :param lb: lower bound
    :param ub: upper bound
    :param nobj: the dimension of objective space (default = 3)
    :param eta_c: spread factor distribution index (default = 20)
    :param eta_m: perturbance factor distribution index (default = 20)
    :param ns: sample number in HV calculation (default = 10000)
    :return:
    """
    # Step 1. Initialization
    nvar = len(lb)  # the dimension of decision space
    pop = np.random.uniform(lb, ub, (npop, nvar))  # population
    objs = cal_obj(pop, nobj)  # the objectives of population
    ref = np.max(objs, axis=0) * 1.2  # reference point

    # Step 2. The main loop
    for t in range(iter):

        if (t + 1) % 50 == 0:
            print('Iteration ' + str(t + 1) + ' completed.')

        # Step 2.1. Mating selection + crossover + mutation
        HV = calHV(objs, ref, npop, ns)  # hypervolume
        mating_pool = selection(pop, HV)  # mating pool
        off = crossover(mating_pool, lb, ub, eta_c)
        off = mutation(off, lb, ub, eta_m)
        off_objs = cal_obj(off, nobj)

        # Step 2.2. Environmental selection
        pop, objs = environmental_selection(np.concatenate((pop, off), axis=0), np.concatenate((objs, off_objs), axis=0), npop, ref, ns)

    # Step 3. Sort the results
    dom = np.full(npop, False)
    for i in range(npop - 1):
        for j in range(i, npop):
            if not dom[i] and dominates(objs[j], objs[i]):
                dom[i] = True
            if not dom[j] and dominates(objs[i], objs[j]):
                dom[j] = True
    pf = objs[~dom]
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.view_init(45, 45)
    x = [o[0] for o in pf]
    y = [o[1] for o in pf]
    z = [o[2] for o in pf]
    ax.scatter(x, y, z, color='red')
    ax.set_xlabel('objective 1')
    ax.set_ylabel('objective 2')
    ax.set_zlabel('objective 3')
    plt.title('The Pareto front of DTLZ1')
    plt.savefig('Pareto front')
    plt.show()


if __name__ == '__main__':
    main(50, 200, np.array([0] * 100), np.array([1] * 100))
