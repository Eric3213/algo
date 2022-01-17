from collections import Counter, defaultdict
from typing import List


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


def maxDepth(s: str):
    ans, temp = 0, 0
    for ch in s:
        if ch == ")":
            temp -= 1
        elif ch == "(":
            temp += 1
            ans = max(ans, temp)
    return ans


def validIPAddress(queryIP: str):
    if queryIP.count(".") == 3:
        nums = queryIP.split(".")
        for x in nums:
            if len(x) == 0 or len(x) > 3:
                return "Neither"
            if (x[0] == '0' and len(x) != 1) or not x.isdigit() or int(x) > 255:
                return "Neither"
        return "IPv4"
    elif queryIP.count(":") == 7:
        nums = queryIP.split(":")
        hexdigits = "0123456789abcdefABCDEF"
        for x in nums:
            if len(x) == 0 or len(x) > 4 or not all(c in hexdigits for c in x):
                return "Neither"
        return "IPv6"
    else:
        return "Neither"


def wordCount(startWords: List[str], targetWords: List[str]) -> int:
    mymap = defaultdict(list)
    for start in startWords:
        n = len(start)
        mymap[n].append(start)

    ans = 0
    for target in targetWords:
        n = len(target)
        target_list = mymap[n - 1]
        flag = 0
        for t in target_list:
            if flag == 1:
                break
            temp = 0
            for ch in t:
                if ch not in target:
                    temp += 1
            if temp == n - 1:
                flag = 1
                ans += 1
    return ans


def isAdditiveNumber(num: str):
    """
    判断一个字符串是否是累加数
    :param num:
    :return:
    """
    n = len(num)
    for secondStart in range(1, n-1):
        if num[0] == "0" and secondStart != 1:
            break
        for secondEnd in range(secondStart, n-1):
            if num[secondStart] == "0" and secondEnd != secondStart:
                break
            if valid(secondStart, secondEnd, num):
                return True
    return False


def valid(secondStart: int, secondEnd: int, num: str):
    n = len(num)
    firstStart, firstEnd = 0, secondStart - 1
    while secondEnd <= n-1:
        third = stringAdd(num, firstStart, firstEnd, secondStart, secondEnd)
        thirdStart = secondEnd + 1
        thirdEnd = secondEnd + len(third)
        if thirdEnd >= n or num[thirdStart:thirdEnd+1] != third:
            break
        if thirdEnd == n-1:
            return True
        firstStart, firstEnd = secondStart, secondEnd
        secondStart, secondEnd = thirdStart, thirdEnd
    return False


def stringAdd(num: str, firstStart: int, firstEnd: int, secondStart: int, secondEnd: int):
    third = []
    carry, cur = 0, 0
    while firstEnd >= firstStart or secondEnd >= secondStart or carry != 0:
        cur = carry
        if firstEnd >= firstStart:
            cur += int(num[firstEnd])
            firstEnd -= 1
        if secondEnd >= secondStart:
            cur += int(num[secondEnd])
            secondEnd -= 1
        carry = cur // 10
        cur %= 10
        third.append(str(cur))
    return "".join(third[::-1])






if __name__ == "__main__":
    # s = "owoztneoer"
    # print(originalDigits(s))
    # s = "how old are you get away fuck you son of a bitch"
    # print(truncateSentence(s, 5))
    # s = '?cbfda?dfad?'
    # print(modifyString(s))
    s = ['ab']
    t = ['abc']
    print(wordCount(s, t))
    # a = [1, 0 , 1, 1, 0, 0, 1]
    # print(minSwaps(a))
