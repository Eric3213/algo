from collections import defaultdict
from typing import List


def maximumInvitations(fa: List[int]) -> int:
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
            print(f"cur={cur}")
            node = graph[cur]
            for w in node:
                print(f"w={w}")
                if w not in seen:
                    total.append(w)
                    q.append(w)
                    seen.add(w)
        return total[-1]

    s = dfs(graph, 0)
    print(s)
    total = []
    dfs(graph, s)
    print(total)
    return len(total)

fa = [2, 2, 1, 2]
print(maximumInvitations(fa))