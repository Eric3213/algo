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


def modifyString(s: str) -> str:
    temp = list(s)
    n = len(temp)
    for i in range(n):
        if temp[i] == '?':
            for change in "abc":
                if not (i > 0 and temp[i - 1] == change or i < n - 1 and temp[i + 1] == change):
                    temp[i] = change
                    break
    return "".join(temp)


def calculate(s: str):
    ops = [1]
    sign = 1
    n = len(s)
    ans = 0
    i = 0
    while i < n:
        if s[i] == ' ':
            i += 1
        elif s[i] == '+':
            sign = ops[-1]
            i += 1
        elif s[i] == '-':
            sign = -ops[-1]
            i += 1
        elif s[i] == '(':
            ops.append(sign)
            i += 1
        elif s[i] == ')':
            ops.pop()
            i += 1
        else:
            num = 0
            while i < n and s[i].isdigit():
                num = num * 10 + ord(s[i]) - ord('0')
                i += 1
            ans += num * sign
    return ans


def numDecoding(s: str):
    n = len(s)
    # f = [1] + [0] * n
    # for i in range(1, n+1):
    #     if s[i-1] != '0':
    #         f[i] += f[i-1]
    #     if i > 1 and s[i-2] != '0' and int(s[i-2:i]) <= 26:
    #         f[i] += f[i-2]
    # return f[n]
    a, b, c = 0, 1, 0
    for i in range(1, n + 1):
        c = 0
        if s[i - 1] != '0':
            c += b
        if i > 1 and s[i - 2] != '0' and int(s[i - 2:i]) <= 26:
            c += a
        a, b = b, c
    return c


def simplifyPath(path: str):
    names = path.split("/")
    stack = list()
    for name in names:
        if name == "..":
            if stack:
                stack.pop()
        elif name and name != ".":
            stack.append(name)
    return "/" + "/".join(stack)


def longestValidParentheses(s: str):
    """
    方法一：动态规划
    方法二：栈
    方法三：降低空间复杂度
    :param s:
    :return:
    """
    # n = len(s)
    # dp = [0] * n
    # if n < 2:
    #     return 0
    # for i in range(1, n):
    #     if s[i] == ')':
    #         if s[i-1] == '(':
    #             dp[i] = dp[i-2] + 2 if i > 1 else 2
    #         else:
    #             if i - dp[i-1] > 0 and s[i - dp[i-1] - 1] == '(':
    #                 dp[i] = dp[i-1] + 2 if i - dp[i-1] - 2 < 0 else dp[i-1] + dp[i - dp[i-1] - 2] + 2
    # return max(dp)

    # 方法二
    # n = len(s)
    # stack = [-1]
    # ans = 0
    # for i in range(n):
    #     if s[i] == "(":
    #         stack.append(i)
    #     else:
    #         stack.pop()
    #         if not stack:
    #             stack.append(i)
    #         else:
    #             tempLength = i - stack[-1]
    #             ans = max(tempLength, ans)
    # return ans

    # 方法三 贪
    n = len(s)
    left, right = 0, 0
    maxLength = 0
    for i in range(n):
        if s[i] == "(":
            left += 1
        else:
            right += 1
        if right == left:
            maxLength = max(maxLength, right * 2)
        elif right > left:
            left, right = 0, 0
    left, right = 0, 0
    for i in range(n-1, -1, -1):
        if s[i] == "(":
            left += 1
        else:
            right += 1
        if right == left:
            maxLength = max(maxLength, right * 2)
        elif left > right:
            left, right = 0, 0
    return maxLength


def lengthOfLongestSubstring(s:str):
    """

    :param s:
    :return: 最长无重复子序列
    """
    n = len(s)
    r, ans = -1, 0
    occ = set()
    for i in range(n):
        if i != 0:
            occ.remove(s[i-1])
        while r + 1 < n and s[r+1] not in occ:
            occ.add(s[r+1])
            r += 1
        ans = max(ans, r - i + 1)
    return ans




if __name__ == "__main__":
    # s = "owoztneoer"
    # print(originalDigits(s))
    # s = "how old are you get away fuck you son of a bitch"
    # print(truncateSentence(s, 5))
    s = '?cbfda?dfad?'
    print(modifyString(s))
