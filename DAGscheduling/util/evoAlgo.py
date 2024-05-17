import math
import pickle
from random import Random
from time import time
import inspyred
#import cPickle as pickle


def get_info(pool, workflow):
    task_names = workflow.topological_sort()
    type_names = pool.info.keys()
    preds = [
        [task_names.index(tn) for tn in workflow.pred(t)] for t in task_names
    ]
    return task_names, type_names, preds


def get_solution(task2node, node2type, task_names, type_names, preds, wf, pool):
    nodes = {}
    aft = [-1] * len(task_names)
    for task in range(len(task_names)):
        node = task2node[task]
        node_type = type_names[node2type[node]]
        comp_time = wf.get_cost(task_names[task]) / pool.info[node_type]["ecu"]
        aft[task] = comp_time
        for t in preds[task]:
            if task2node[t] == task2node[task]:
                comm_time = 0
            else:
                n1_t = type_names[node2type[task2node[t]]]
                speed = min([pool.info[n1_t]["net"], pool.info[node_type]["net"]])
                comm_time = wf.get_data_size(task_names[t], task_names[task]) / speed
            est = comm_time + aft[t]
            eft = comp_time + est
            if aft[task] < 0 or aft[task] < eft:
                aft[task] = eft
        if node not in nodes:
            nodes[node] = {
                "start": aft[task] - comp_time,
                "end": aft[task],
                "events": [(task, aft[task] - comp_time, aft[task])]
            }
        else:
            if nodes[node]["end"] > aft[task] - comp_time:
                aft[task] = nodes[node]["end"] + comp_time
            nodes[node]["end"] = aft[task]
            nodes[node]["events"].append((task, aft[task] - comp_time, aft[task]))
    return aft, nodes


def charge(nodes, pool, node2type, type_names):
    return sum([
        math.ceil((v["end"] - v["start"]) / 3600) * pool.info[type_names[node2type[n]]]["price"] \
        for n, v in nodes.iteritems()
    ])


def evaluate_individual(candidate, task_names, type_names, preds, wf, pool):
    task2node, node2type = candidate
    aft, nodes = get_solution(task2node, node2type, task_names, type_names, preds, wf, pool)
    o1 = aft[len(task2node) - 1]
    o2 = charge(nodes, pool, node2type, type_names)
    return round(o1, 0), round(o2, 2)


@inspyred.ec.evaluators.evaluator
def evaluator(candidate, args):
    task_names = args.get("task_names")
    type_names = args.get("type_names")
    preds = args.get("preds")
    wf = args.get("wf")
    pool = args.get("pool")
    return inspyred.ec.emo.Pareto(evaluate_individual(candidate, task_names, type_names, preds, wf, pool))


def generator(random, args):
    num_task = len(args.get("task_names"))
    num_type = len(args.get("type_names"))
    task2node = [random.randint(0, num_task - 1) for _ in range(num_task)]
    node2type = [random.randint(0, num_type - 1) for _ in range(num_task)]
    return task2node, node2type


@inspyred.ec.variators.crossover
def cross(random, mom, dad, args):
    m1, m2 = mom
    d1, d2 = dad
    loc1 = random.randint(0, len(m1) - 1)
    loc2 = random.randint(0, len(m2) - 1)
    return [(m1[:loc1] + d1[loc1:], m2[:loc2] + d2[loc2:])]


@inspyred.ec.variators.mutator
def mutate(random, candidate, args):
    task2node, node2type = candidate
    num_task = len(args.get("task_names"))
    num_type = len(args.get("type_names"))
    t1 = task2node[:]
    t2 = node2type[:]
    for i in range(num_task):
        p = random.random()
        if p <= 1.0 / num_task:
            t1[i] = random.randint(0, num_task - 1)
        p = random.random()
        if p <= 1.0 / num_task:
            t2[i] = random.randint(0, num_type - 1)
    return t1, t2


def nsga2(task_names, type_names, preds, wf, pool):
    prng = Random()
    prng.seed(time())

    ea = inspyred.ec.emo.NSGA2(prng)
    ea.variator = [cross, mutate]
    ea.terminator = inspyred.ec.terminators.generation_termination
    # ea.observer = inspyred.ec.observers.plot_observer
    final_pop = ea.evolve(
        generator=generator,
        evaluator=evaluator,
        # evaluator=inspyred.ec.evaluators.parallel_evaluation_pp,
        # pp_evaluator=evaluator,
        pop_size=10,
        maximize=False,
        max_generations=1000,
        type_names=type_names,
        task_names=task_names,
        preds=preds,
        wf=wf,
        pool=pool)
    s = set()
    ar = []
    for i in ea.archive:
        if tuple(i.fitness) not in s:
            s.add(tuple(i.fitness))
            ar.append(i)
    return ar


# return ea.archive

def save(filename, ar, task_names, type_names, preds, wf, pool):
    l = []
    for i in ar:
        a = []
        task2node, node2type = i.candidate
        _, nodes = get_solution(task2node, node2type, task_names, type_names, preds, wf, pool)
        for n, v in nodes.iteritems():
            t = type_names[node2type[n]]
            es = [(task_names[task], start, end) for task, start, end in v["events"]]
            a.append((n, t, es))
        l.append((i.fitness[0], i.fitness[1], a))
    pickle.dump(l, open(filename, "w"))


if __name__ == '__main__':
    from sys import argv
    from workflow import Workflow
    from pool import AWS
    #import hotshot, hotshot.stats

    wf = Workflow(argv[1])
    pool = AWS("aws.info")
    task_names, type_names, preds = get_info(pool, wf)
    archive = nsga2(task_names, type_names, preds, wf, pool)

    # prof = hotshot.Profile("emo2.prof")
    # archive = prof.runcall(nsga2, task_names, type_names, preds, wf, pool)
    # prof.close()
    # prof.close()
    # stats = hotshot.stats.load("emo2.prof")
    # stats.strip_dirs()
    # stats.sort_stats('cumtime', 'calls')
    # stats.print_stats()

    archive = sorted(archive, key=lambda x: x.fitness[0], reverse=True)
    for i in range(len(archive)):
        print("[%d]: Makespan: %.0fs\tCost: $%.2f" % (i, archive[i].fitness[0], archive[i].fitness[1]))

    save("%s.ar" % argv[1][:-4], archive, task_names, type_names, preds, wf, pool)

# rand = Random()
# rand.seed(int(time()))

# task_names, type_names, preds = get_info(cloud, wf)
# candidate = (range(len(wf)), [1]*len(wf))
# print evaluate_individual(candidate, task_names, type_names, preds, wf, cloud)
# candidate = ([1]*len(wf), [1]*len(wf))
# print evaluate_individual(candidate, task_names, type_names, preds, wf, cloud)
# for _ in range(10):
# 	candidate = generate(rand, {"num_task":len(task_names), "num_type":len(type_names)})
# 	print evaluate_individual(candidate, task_names, type_names, preds, wf, cloud)
