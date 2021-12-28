from collections import Counter


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


def truncateSentence(s: str, k: int) -> str:
    return " ".join(s.split(" ")[:k])


if __name__ == "__main__":
    # s = "owoztneoer"
    # print(originalDigits(s))
    s = "how old are you get away fuck you son of a bitch"
    print(truncateSentence(s, 5))
