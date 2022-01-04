from collections import defaultdict, deque
from functools import lru_cache
from typing import List


def maximumInvitations0(fa: List[int]) -> int:
    n = len(fa)
    graph = defaultdict(list)
    for i, f in enumerate(fa):
        graph[i].append(f)
        graph[f].append(i)

    total = []

    def dfs(graph, s):
        q = []
        q.append(s)
        total.append(s)
        seen = set()
        seen.add(s)
        while (len(q) > 0):
            cur = q.pop()
            # print(f"cur={cur}")
            node = graph[cur]
            for w in node:
                # print(f"w={w}")
                if w not in seen:
                    total.append(w)
                    q.append(w)
                    seen.add(w)
        return total[-1]

    s = dfs(graph, 0)
    # print(s)
    total = []
    dfs(graph, s)
    # print(total)
    return len(total)


def maximumInvitations(graph: List[int]):
    n = len(graph)
    # 入度
    degree = [0] * n
    for w in graph:
        degree[w] += 1

    # 拓扑排序，同时得到最长链 deque
    max_depth = [0] * n
    q = deque(i for i, d in enumerate(degree) if d == 0)
    while q:
        v = q.popleft()
        max_depth[v] += 1
        w = graph[v]
        max_depth[w] = max(max_depth[w], max_depth[v])
        degree[w] -= 1
        if degree[w] == 0:
            q.append(w)

    # maxRingSize sumChainSize 遍历deg
    maxRingSize, sumChainSize = 0, 0
    for i, d in enumerate(degree):
        if d == 0:
            continue
        degree[i] = 0
        ring_size = 1
        v = graph[i]
        while v != i:
            degree[v] = 0
            ring_size += 1
            v = graph[v]
        if ring_size == 2:
            sumChainSize += 2 + max_depth[i] + max_depth[graph[i]]
        else:
            maxRingSize = max(maxRingSize, ring_size)
    return max(maxRingSize, sumChainSize)


def catMouseGame(graph: List[List[int]]):
    @lru_cache(None)
    def dfs(m, c, i):
        if i > len(graph) * 2:
            return 0
        if m == 0:
            return -1
        if m == c:
            return 1
        res = (-1) ** i
        if i % 2:
            for nxt in graph[c]:
                if nxt:
                    res = max(res, dfs(m, nxt, i+1))
                if res == 1:
                    break
        else:
            for nxt in graph[m]:
                res = min(res, dfs(nxt, c, i+1))
                if res == -1:
                    break
        return res
    ans = dfs(1, 2, 0)
    return 0 if ans == 0 else (1 if ans == -1 else 2)


fa = [2, 2, 1, 2]
print(maximumInvitations(fa))
