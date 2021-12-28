from collections import defaultdict, Counter
from typing import List


class RangeFreqQuery:

    def __init__(self, arr: List[int]):
        self.d = defaultdict(lambda: defaultdict(int))
        d = self.d
        n = len(arr)
        temp = defaultdict(int)
        for i in range(n):
            temp[arr[i]] += 1
            d[i] = temp

    def query(self, left: int, right: int, value: int) -> int:
        d = self.d
        if left == 0:
            return d[right][value]
        else:
            return d[right][value] - d[left - 1][value]


def originalDigits(s: str) -> str:
    d = Counter(s)
    cnt = [0] * 10
    cnt[0] = d['z']
    cnt[2] = d['w']
    cnt[4] = d['u']
    cnt[6] = d['x']
    cnt[8] = d['g']

    cnt[3] = d['h'] - cnt[8]
    cnt[5] = d['f'] - cnt[4]
    cnt[7] = d['s'] - cnt[6]
    cnt[1] = d['o'] - cnt[0] - cnt[2] - cnt[4]
    cnt[9] = d['i'] - cnt[5] - cnt[6] - cnt[8]

    return "".join(str(x) * cnt[x] for x in range(10))


if __name__ == "__main__":
    s = "owoztneoer";
    print(originalDigits(s))

# newr = RangeFreqQuery([12,33,4,56,22,2,34,33,22,12,34,56])
# print(newr.query(1,2,4))
# print(newr.query(0,11,33))
