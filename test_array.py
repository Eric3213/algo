import bisect
from collections import Counter, defaultdict, deque
from functools import cmp_to_key, lru_cache
from typing import List
import math
import heapq
import random


def maxArea(height: List[int]):
    l, r = 0, len(height) - 1
    ans = 0
    while l < r:
        area = min(height[l], height[r]) * (r - l)
        ans = max(ans, area)
        if height[l] <= height[r]:
            l += 1
        else:
            r -= 1
    return ans


def threeSum(nums):
    n = len(nums)
    nums.sort()
    ans = []
    for first in range(n):
        if first > 0 and nums[first] == nums[first - 1]:
            continue
        target = -nums[first]
        third = n - 1
        for second in range(first + 1, n):
            if second > first + 1 and nums[second] == nums[second - 1]:
                continue
            while second < third and nums[second] + nums[third] > target:
                third -= 1
            if second == third:
                break
            if nums[second] + nums[third] == target:
                ans.append([nums[first], nums[second], nums[third]])
    return ans


def findAllPeople(n: int, m: List[List[int]], f: int) -> List[int]:
    m = sorted(m, key=(lambda x: [x[2], x[0]]))
    n = len(m)
    ans = []
    ans.append(0)
    tempt = m[0][2]
    x, y = m[0][0], m[0][1]
    if x == f or y == f:
        ans.append(f)
    if x in ans and y not in ans:
        ans.append(y)
    if x not in ans and y in ans:
        ans.append(x)
    temp = []
    for i in range(1, n):
        x, y, t = m[i][0], m[i][1], m[i][2]

        if x == f or y == f:
            if f not in ans:
                ans.append(f)
        if t == tempt:

            if x in ans and y in ans:
                continue
            elif x in ans:
                temp.append(y)
            elif y in ans:
                temp.append(x)
        else:
            # temp bingru ans
            for t in temp:
                if t not in ans:
                    ans.append(t)
            temp = []
            if x in ans and y in ans:
                continue
            elif x in ans:
                temp.append(y)
            elif y in ans:
                temp.append(x)
        tempt = t
    for t in temp:
        if t not in ans:
            ans.append(t)
    if f not in ans:
        ans.append(f)
    return ans


def kthSmallestPrimeFraction(arr: List[int], k: int) -> List[int]:
    def cmp(o1, o2):
        return -1 if o1[0] * o2[1] < o1[1] * o2[0] else 1

    n = len(arr)
    frac = list()
    for i in range(n):
        for j in range(i + 1, n):
            frac.append((arr[i], arr[j]))
    frac.sort(key=cmp_to_key(cmp))
    return list(frac[k - 1])


def superPow(a: int, b: List[int]) -> int:
    MOD = 1337
    ans = 1
    for e in reversed(b):
        ans = ans * pow(a, e, MOD) % MOD
        a = pow(a, 10, MOD)
    return ans


def findEvenNumbers(digits: List[int]) -> List[int]:
    ans = []
    freq = Counter(digits)
    for i in range(100, 1000, 2):
        frei = Counter([int(d) for d in str(i)])
        if all(freq[d] - frei[d] >= 0 for d in frei.keys()):
            ans.append(i)
    return ans


def maxSumOfThreeSubarrays(nums: List[int], k: int):
    ans = []
    sum1, maxSum1, maxSum1Idx = 0, 0, 0
    sum2, maxSum2, maxSum12Idx = 0, 0, ()
    sum3, totalSum = 0, 0
    for i in range(2 * k, len(nums)):
        sum1 += nums[i - 2 * k]
        sum2 += nums[i - k]
        sum3 += nums[i]
        if i >= 3 * k - 1:
            if sum1 > maxSum1:
                maxSum1 = sum1
                maxSum1Idx = i - 3 * k + 1
            if maxSum1 + sum2 > maxSum2:
                maxSum2 = maxSum1 + sum2
                maxSum12Idx = (maxSum1Idx, i - 2 * k + 1)
            if maxSum2 + sum3 > totalSum:
                totalSum = maxSum2 + sum3
                ans = [*maxSum12Idx, i - k + 1]
            sum1 -= nums[i - 3 * k + 1]
            sum2 -= nums[i - 2 * k + 1]
            sum3 -= nums[i - k + 1]
    return ans


def countPoints(rings: str) -> int:
    n = len(rings) // 2

    temp = [[0] * 3 for _ in range(10)]
    i = 0
    ans = []
    while i < n * 2:
        a = int(rings[i + 1])
        if rings[i] == 'R':
            temp[a][0] = 1
        elif rings[i] == 'G':
            temp[a][1] = 1
        else:
            temp[a][2] = 1
        if temp[a][0] == 1 and temp[a][1] == 1 and temp[a][2] == 1:
            ans.append(a)
        i += 2
    return len(set(ans))


def maxTotalFruits(fruits: List[List[int]], startPos: int, k: int) -> int:
    p = []
    for po, a in fruits:
        p.append(po)
    n = len(p)
    i, j = bisect.bisect_left(p, startPos), bisect.bisect_left(p, startPos + k)
    if j == n:
        j -= 1
    ans = 0
    for k in range(i, j + 1):
        ans += fruits[k][1]
    print(f"?????????ans??? {ans}")
    while p[j] > startPos:
        print(f"p[j] = {p[j]}")

        temp = p[j] - startPos
        if temp > k // 2:
            j -= 1
            continue
        temp = startPos - k + temp * 2
        s = bisect.bisect_left(p, temp)
        print(f"s = {s} and j = {j}")
        cur = 0
        for k in range(s, j + 1):
            cur += fruits[k][1]
        ans = max(ans, cur)

        j -= 1
    i, j = bisect.bisect_left(p, startPos - k), bisect.bisect_right(p, startPos)
    cur = 0
    for k in range(i, j):
        cur += fruits[k][1]

    return max(ans, cur)


def getDescentPeriods(prices: List[int]) -> int:
    n = len(prices)
    ans = 1
    temp = []
    length = 1
    for i in range(1, n):
        if prices[i] == prices[i - 1] - 1:
            length += 1
        else:
            temp.append(length)
            length = 1
    print(temp)
    for i in range(len(temp)):
        ans += (temp[i] + 1) * temp[i] // 2
    return ans


def kIncreasing(arr: List[int], k: int) -> int:
    def getHeight(men):
        longest = {}  # c???????????????
        longest[0] = 1
        for i in range(1, len(men)):
            maxlen = -1
            for j in range(0, i):
                if men[i] > men[j] and maxlen < longest[j]:
                    maxlen = longest[j]
            if maxlen >= 1:  # ????????????????????????????????????ax????????????
                longest[i] = maxlen + 1
            else:
                longest[i] = 1
        return max(longest.values())

    a = []
    ans = 0
    n = len(arr)
    for i in range(k):
        temp = []
        j = i
        while j < n:
            temp.append(arr[j])
            j += k
        a.append(temp[:])
    print(a)
    for temp in a:
        ans += getHeight(temp)
    return ans


def rotate(matrix: List[List[int]]):
    n = len(matrix)
    for i in range(n // 2):
        for j in range(n):
            matrix[i][j], matrix[n - i - 1][j] = matrix[n - i - 1][j], matrix[i][j]
    for i in range(n):
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


def dayOfTheWeek(day: int, month: int, year: int):
    week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    monthdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    ans = 0
    ans += 365 * (year - 1971) + (year - 1969) // 4
    ans += sum(monthdays[:month - 1])
    if month >= 3 and (year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)):
        ans += 1
    ans += day
    return week[(ans + 3) % 7]


def twoSum(nums: List[int], target: int):
    temp = defaultdict()
    for i, n in enumerate(nums):
        if target - n in temp:
            return [i, temp[target - n]]
        else:
            temp[n] = i
    return []


def grayCode(n: int):
    """

    :param n:
    :return: n????????????
    """
    ans = [0]
    for i in range(1, n + 1):
        for j in range(len(ans) - 1, -1, -1):
            ans.append(ans[j] | (1 << (i - 1)))
    return ans


def combinationSum(candidates: List[int], target: int):
    """
    ??????+??????
    :param candidates: ???????????????
    :param target: ????????????
    :return: ?????????????????????????????????
    """
    candidates.sort()
    ans = []

    def find(start, use, remain):
        for i in range(start, len(candidates)):
            c = candidates[i]
            if c == remain:
                ans.append(use + [c])
            elif c < remain:
                find(i, use + [c], remain - c)
            else:
                return

    find(0, [], target)
    return ans


def permuteUnique(nums: List[int]):
    """
    ???????????? ??????????????????
    :param nums: ?????????????????????????????????
    :return: ???????????????????????????
    """
    ans = []

    def dfs(nums, path):
        if not nums:
            ans.append(path)
            return
        temp = list(set(nums))
        for idx, i in enumerate(temp):
            # ????????????????????????
            nums.remove(i)
            # print(f"remove???{nums}")
            dfs(nums, path + [i])
            nums.append(i)
            # print(f"append???{nums}")

    dfs(nums, [])
    return ans


def minSwaps(nums: List[int]):
    n = len(nums)
    m = nums.count(1)
    j, i = 0, n - m + 1
    cur = 0
    if nums[0] == 1:
        cur += 1
    while i < n:
        if nums[i] == 1:
            cur += 1
        i += 1
    i = n - m + 1
    ans = cur
    while i + 1 < n:
        i += 1
        j += 1
        if nums[j] == 1:
            cur += 1
        if nums[i - 1] == 1:
            cur -= 1
        ans = max(ans, cur)
    cur = nums[:m].count(1)
    ans = max(ans, cur)
    i, j = 0, m - 1
    while j + 1 < n:
        i += 1
        j += 1
        if nums[i - 1] == 1:
            cur -= 1
        if nums[j] == 1:
            cur += 1
        ans = max(ans, cur)
    return m - ans


def dominantIndex(nums: List[int]):
    m1, m2, idx = -1, -1, 0
    for i, n in enumerate(nums):
        if n > m1:
            m1, m2, idx = n, m1, i
        elif n > m2:
            m2 = n
    return idx if m1 >= m2 * 2 else -1
    # n = len(nums)
    # if n == 1:
    #     return 0
    # if nums[0] >= nums[1]:
    #     largest, secondLarge = nums[0], nums[1]
    #     largestIndex = 0
    # else:
    #     largest, secondLarge = nums[1], nums[0]
    #     largestIndex = 1
    # for i in range(2, n):
    #     if nums[i] > largest:
    #         secondLarge = largest
    #         largest = nums[i]
    #         largestIndex = i
    #     elif nums[i] > secondLarge:
    #         secondLarge = nums[i]
    # print(largest)
    # print(secondLarge)
    # if largest // 2 >= secondLarge:
    #     return largestIndex
    # else:
    #     return -1


def containsNearbyDuplicate(nums: List[int], k: int):
    dic = defaultdict(int)
    for i, n in enumerate(nums):
        if dic.get(n) is not None:
            if i - dic.get(n) <= k:
                return True
        dic[n] = i
    return False


class StockPrice:

    def __init__(self):
        self.price = SortedList()
        self.maxTimeStamp = 0
        self.timePriceMap = defaultdict(int)

    def update(self, timestamp: int, price: int) -> None:
        if timestamp in self.timePriceMap:
            self.price.discard(self.timePriceMap[timestamp])
        self.price.add(price)
        self.maxTimeStamp = max(self.maxTimeStamp, timestamp)
        self.timePriceMap[timestamp] = price

    def current(self):
        return self.timePriceMap[self.maxTimeStamp]

    def maximum(self):
        return self.price[-1]

    def minimum(self):
        return self.price[0]


class DetectSquares:

    def __init__(self):
        self.d = defaultdict(int)
        self.s = set()

    def add(self, point: List[int]):
        self.d[(point[0], point[1])] += 1
        self.s.add((point[0], point[1]))

    def count(self, point: List[int]):
        ans = 0
        for pre in self.s:
            if pre[0] == point[0] and pre[1] == point[1]: continue
            if abs(pre[0] - point[0]) == abs(pre[1] - point[1]):
                ans += 1 * self.d[(pre[0], pre[1])] * self.d[(pre[0], point[1])] * self.d[(point[0], pre[1])]
        return ans


def numberOfWeakCharacters(properties: List[List[int]]):
    properties.sort(key=lambda x: (-x[0], x[1]))
    ans = 0
    maxDef = 0
    for _, def_ in properties:
        if maxDef > def_:
            ans += 1
        else:
            maxDef = max(maxDef, def_)
    return ans


def highestPeak(isWater: List[List[int]]):
    m, n = len(isWater), len(isWater[0])
    ans = [[water - 1 for water in row] for row in isWater]
    q = deque((i, j) for i, row in enumerate(isWater) for j, water in enumerate(row) if water)
    while q:
        i, j = q.popleft()
        for x, y in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= x < m and 0 <= y < n and ans[x][y] == -1:
                ans[x][y] = ans[i][j] + 1
                q.append((x, y))
    return ans


def findMinFibonacciNumbers(k: int):
    f = [1, 1]
    while f[-1] < k:
        f.append(f[-1] + f[-2])
    ans, i = 0, len(f) - 1
    while k:
        if k >= f[i]:
            k -= f[i]
            ans += 1
        i -= 1
    return ans


def getMaximumGold(grid: List[List[int]]):
    m, n = len(grid), len(grid[0])
    ans = 0

    def dfs(x: int, y: int, gold: int):
        gold += grid[x][y]
        nonlocal ans
        ans = max(ans, gold)
        rec = grid[x][y]
        grid[x][y] = 0

        for nx, ny in ((x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] > 0:
                dfs(nx, ny, gold)

        grid[x][y] = rec

    for i in range(m):
        for j in range(n):
            if grid[i][j] != 0:
                dfs(i, j, 0)
    return ans


def sumOfUnique(nums: List[int]):
    return sum(num for num, cnt in Counter(nums).items() if cnt == 1)
    # counter = Counter(nums)
    # ans = 0
    # for num, cnt in counter.items():
    #     if cnt == 1:
    #         ans += num
    # return ans


def countKDifference(nums: List[int], k: int):
    ans = 0
    cnt = Counter()
    for num in nums:
        ans += cnt[num - k] + cnt[num + k]
        cnt[num] += 1
    return ans


def simplifiedFractions(n: int) -> List[str]:
    ans = []
    for denominetor in range(2, n + 1):
        for numerator in range(1, denominetor):
            if math.gcd(denominetor, numerator) == 1:
                ans.append(f"{numerator}/{denominetor}")
    return ans
    # return [f"{numerator}/{denominator}" for denominator in range(2, n + 1) for numerator in range(1, denominator) if gcd(denominator, numerator) == 1]


def numEnclaves(grid: List[List[int]]):
    """
    :param grid:
    :return: ???????????????
    """
    m, n = len(grid), len(grid[0])
    vis = [[False] * n for _ in range(m)]

    def dfs(r: int, c: int):
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] == 0 or vis[r][c]:
            return
        vis[r][c] = True
        for x, y in ((r - 1, c), (r + 1, c), (r, c + 1), (r, c - 1)):
            dfs(x, y)

    for i in range(m):
        dfs(i, 0)
        dfs(i, n - 1)
    for i in range(1, n - 1):
        dfs(0, i)
        dfs(m - 1, i)
    return sum(grid[i][j] and not vis[i][j] for i in range(m) for j in range(n))


def singleNonDuplicate(nums: List[int]):
    """

    :param nums: ?????????
    :return: ????????????????????????  ??????log(n)??????????????? O(1)???????????????
    """
    low, high = 0, len(nums) - 1
    while low < high:
        mid = (low + high) // 2
        if nums[mid] == nums[mid ^ 1]:
            low = mid + 1
        else:
            high = mid
    return nums[low]


def luckyNumbers(matrix: List[List[int]]):
    minRow = [min(row) for row in matrix]
    minCol = [max(col) for col in zip(*matrix)]
    ans = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == minRow[i] == minCol[j]:
                ans.append(matrix[i][j])
    return ans


def knightProbability(n: int, k: int, row: int, column: int):
    dp = [[[0] * n for _ in range(n)] for _ in range(k + 1)]
    for step in range(k + 1):
        for i in range(n):
            for j in range(n):
                if step == 0:
                    dp[step][i][j] = 1
                else:
                    for di, dj in ((-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n:
                            dp[step][i][j] += dp[step - 1][ni][nj] / 8
    return dp[k][row][column]


def pancakeSort(arr: List[int]):
    """
    ????????????
    :param arr:
    :return:
    """
    ans = []
    for n in range(len(arr), 1, -1):
        index = 0
        for i in range(n):
            if arr[i] > arr[index]:
                index = i
        if index == n - 1:
            continue
        m = index
        for i in range((m + 1) // 2):
            arr[i], arr[m - i] = arr[m - i], arr[i]
        for i in range(n // 2):
            arr[i], arr[n - 1 - i] = arr[n - 1 - i], arr[i]
        ans.append(index + 1)
        ans.append(n)
    return ans


def pushDominoes(dominoes: str):
    s = list(dominoes)
    n, i, left = len(s), 0, 'L'
    while i < n:
        j = i
        while j < n and s[j] == '.':
            j += 1
        right = s[j] if j < n else 'R'
        if left == right:
            while i < j:
                s[i] = right
                i += 1
        elif left == 'R' and right == 'L':
            k = j - 1
            while i < k:
                s[k] = right
                s[i] = left
                i += 1
                k -= 1
        left = right
        i = j + 1
    return ''.join(s)


def findBalls(grid: List[List[int]]):
    n = len(grid[0])
    ans = [-1] * n
    for j in range(n):
        col = j
        for row in grid:
            dir = row[col]
            col += dir
            if col < 0 or col == n or row[col] != dir:
                col = -1
                break
        ans[j] = col
    return ans


def findKthLargest(nums: List[int], k: int):
    """
    python????????????????????????
    :param nums:
    :return: ???k????????????
    """

    # h = [x for x in nums[:k]]
    # heapq.heapify(h)
    # n = len(nums)
    # for i in range(k, n):
    #     if nums[i] > h[0]:
    #         heapq.heappop(h)
    #         heapq.heappush(h, nums[i])
    # return h[0]

    def partition(nums: List[int], left: int, right: int):
        pivot = nums[left]
        i, j = left, right
        while i < j:
            while i < j and nums[j] >= pivot:
                j -= 1
            nums[i] = nums[j]
            while i < j and nums[i] <= pivot:
                i += 1
            nums[j] = nums[i]
        nums[i] = pivot
        return i

    def topk_split(nums: List[int], k: int, left: int, right: int):
        if left < right:
            index = partition(nums, left, right)
            if index == k:
                return
            elif index < k:
                topk_split(nums, k, index + 1, right)
            else:
                topk_split(nums, k, left, index - 1)

    topk_split(nums, len(nums) - k, 0, len(nums) - 1)
    return nums[len(nums) - k]


def maxSubArray(nums: List[int]):
    n = len(nums)
    cur, ans = 0, 0
    for i in range(1, n):
        cur = max(nums[i], nums[i] + cur)
        ans = max(ans, cur)
    return ans


def maxProfit(price: List[int]):
    # ????????????
    # min_price = float('-inf')
    # ans = 0
    # for p in price:
    #     ans = max(ans, p - min_price)
    #     min_price = min(min_price, p)
    # return ans

    # ????????????  ???????????????????????????
    # ??????
    # n = len(price)
    # ans = 0
    # for i in range(1, n):
    #     temp = price[i] - price[i-1]
    #     ans += temp if temp > 0 else 0
    # return ans

    # todo ?????????
    pass


def goodDaysToBobBank(security: List[int], time: int):
    n = len(security)
    left = [0] * n
    right = [0] * n
    for i in range(1, n):
        if security[i] <= security[i - 1]:
            left[i] = left[i - 1] + 1
        if security[n - i - 1] <= security[n - i]:
            right[n - i - 1] = right[n - i] + 1
    return [i for i in range(time, n - time) if left[i] >= time and right[i] >= time]


def findRestaurant(list1: List[str], list2: List[str]):
    ans = list()
    d = defaultdict(int)
    for i, n in enumerate(list1):
        d[n] = i
    minIndex = len(list1) + len(list2)
    curIndex = 0
    for i, n in enumerate(list2):
        if n in d:
            curIndex = i + d[n]
            if curIndex < minIndex:
                minIndex = curIndex
                ans = [n]
            elif curIndex == minIndex:
                ans.append(n)
    return ans


def search(nums: List[int], target: int):
    n = len(nums)
    left, right = 0, n
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return -1


def searchInsert(nums: List[int], target: int):
    n = len(nums)
    left, right = 0, n
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left + 1


def searchRange(nums: List[int], target: int):
    """
    :param nums:
    :param target:
    :return: nums???target???????????????range
    """
    n = len(nums)

    def binarySearch(nums: List[int], target: int):
        """
        :param nums:
        :param target:
        :return: ?????? ????????? >= target ?????????
        """
        n = len(nums)
        left, right = 0, n
        while left < right:
            mid = (left + right) // 2
            if nums[mid] >= target:
                right = mid
            else:
                left = mid + 1
        return left

    left = binarySearch(nums, target)
    right = binarySearch(nums, target + 1) - 1
    if left >= n or nums[left] != target:
        return [-1, -1]
    return [left, right]


def findMissingPositive(nums: List[int]):
    n = len(nums)
    for i in range(n):
        if nums[i] <= 0:
            nums[i] = n + 1

    for i in range(n):
        num = nums[i]
        if num <= n:
            nums[num - 1] = -abs(nums[num - 1])

    for i in range(n):
        if nums[i] > 0:
            return i + 1

    return n + 1


def longestWords(words: List[str]):
    longest = ""
    candidates = {""}
    words.sort(key=lambda x: (-len(x), x), reverse=True)
    for word in words:
        if word[:-1] in candidates:
            longest = word
            candidates.add(word)
    return longest


def totalFruit(fruits: List[int]):
    if len(fruits) == 1:
        return 1
    l, r = 0, 1
    ans = 0
    type = [fruits[0]]
    while r < len(fruits):
        if fruits[r] != fruits[r - 1] and fruits[r] not in type:
            ans = max(ans, r - l)
            l = r - 1
            while l > 0 and fruits[l - 1] == fruits[l]:
                l -= 1
            type = [fruits[l], fruits[r]]
        r += 1
    return max(ans, r - l)


def minWin(s: str, t: str):
    """
    :param s:
    :param t:
    :return: s?????????t???????????????????????????
    """
    if len(s) < len(t):
        return ""
    n = len(s)
    t = Counter(t)
    start, end = 0, 0
    min_len = float('inf')
    look_up = Counter()
    ans = ""
    while end < n:
        look_up[s[end]] += 1
        end += 1
        while all(map(lambda x: look_up[x] >= t[x], t.keys())):
            if end - start < min_len:
                ans = s[start: end]
                min_len = end - start
            look_up[s[start]] -= 1
            start += 1
    return ans


def lastStoneWeight(stones: List[int]):
    heap = [-stone for stone in stones]
    heapq.heapify(heap)
    while len(heap) > 1:
        x, y = heapq.heappop(heap), heapq.heappop(heap)
        if x != y:
            heapq.heappush(heap, x - y)
    return 0 if not heap else -heap[0]


def uniqueOccurrences(arr: List[int]):
    c = Counter(arr)
    s = c.values()
    return len(s) == len(set(s))


def trailingZeroes(n: int) -> int:
    if n == 0: return 0
    zeros = [0] * n
    no_zeros = [1] * n
    for i in range(1, n):
        temp = no_zeros[i - 1] * (i + 1)
        addict = 0
        while temp % 10 == 0:
            temp //= 10
            addict += 1
        no_zeros[i] = temp
        zeros[i] = zeros[i - 1] + addict
    return zeros[n - 1]


def generateMatrix(n: int):
    matrix = [[0] * n for _ in range(n)]
    left, right, up, down = 0, n - 1, 0, n - 1
    num = 1
    while left < right and up < down:
        for x in range(left, right):
            matrix[up][x] = num
            num += 1
        for x in range(up, down):
            matrix[x][right] = num
            num += 1
        for x in range(right, left, -1):
            matrix[down][x] = num
            num += 1
        for x in range(down, up, -1):
            matrix[x][left] = num
            num += 1
        left += 1
        right -= 1
        up += 1
        down -= 1
    if n % 2:
        matrix[n // 2][n // 2] = num
    return matrix


def missingRolls(rolls: List[int], mean: int, n: int):
    missingSum = mean * (len(rolls) + n) - sum(rolls)
    if not n <= missingSum <= n * 6:
        return []
    zheng, yu = divmod(missingSum, n)
    return [zheng + 1] * yu + [zheng] * (n - yu)


# def deserialize(s: str):
#     index = 0
#
#     def dfs() -> NestedInteger:
#         nonlocal index
#         if s[index] == '[':
#             index += 1
#             ni = NestedInteger()
#             while s[index] != ']':
#                 ni.add(dfs())
#                 if s[index] == ',':
#                     index += 1
#             index += 1
#             return ni
#         else:
#             negative = False
#             if s[index] == '-':
#                 negative = True
#                 index += 1
#             num = 0
#             while index < len(s) and s[index].isdigit():
#                 num *= 10
#                 num += int(s[index])
#                 index += 1
#             if negative:
#                 num = -num
#             return NestedInteger(num)
#
#     return dfs()


def findKthNumber(self, m: int, n: int, k: int):
    """
    ??????????????????k????????????   ??????????????????
    :param self:
    :param m: m???
    :param n: n???
    :param k: ???k????????????
    :return:
    """
    if m > n:
        return self.findKthNumber(n, m, k)
    left, right = 1, n * m

    def check(num):
        return sum(min(n, num // row) for row in range(1, m + 1))

    while left < right:
        mid = (left + right) >> 1
        if check(mid) < k:
            left = mid + 1
        else:
            right = mid
    return left


def repeatedNTimes(self, nums: List[int]):
    n = len(nums)

    while True:
        x, y = random.randrange(n), random.randrange(n)
        if x != y and nums[x] == nums[y]:
            return nums[x]


def canIWin(maxChoosableInteger: int, desiredTotal: int):
    """
    ?????????????????????100 game
    :param maxChoosableInteger: ???1-maxChoosableInteger
    :param desiredTotal:  ??????
    :return: ???????????????
    """

    @lru_cache
    def dfs(usedNumbers: int, curTotal: int):
        for i in range(maxChoosableInteger):
            if (usedNumbers >> i) & 1 == 0:
                if curTotal + i + 1 >= desiredTotal or not dfs(usedNumbers | (1 << i), curTotal + i + 1):
                    return True
        return False

    return (1 + maxChoosableInteger) * maxChoosableInteger // 2 >= desiredTotal and dfs(0, 0)


class Solution:

    def makeSquare(self, matchsticks: List[int]):
        """
        ????????????  ?????????????????????  ?????? ??????+??????
        :param matchsticks:
        :return:
        """
        totalLen = sum(matchsticks)
        if totalLen % 4:
            return False
        tLen = totalLen // 4

        dp = [-1] * (1 << len(matchsticks))
        dp[0] = 0
        for s in range(1, len(dp)):
            for k, v in enumerate(matchsticks):
                if s & (1 << k) == 0:
                    continue
                s1 = s & ~(1 << k)
                if dp[s1] >= 0 and dp[s1] + v <= tLen:
                    dp[s] = (dp[s1] + v) % tLen
                    break
        return dp[-1] == 0


class MyCalendarThree:
    def __init__(self):
        self.tree = defaultdict(int)
        self.lazy = defaultdict(int)

    def update(self, start: int, end: int, left: int, right: int, idx: int):
        if start > right or end < left:
            return
        if start <= left and end >= right:
            self.tree[idx] += 1
            self.lazy[idx] += 1
        else:
            mid = left + (right - left) // 2
            self.update(start, end, left, mid, idx * 2)
            self.update(start, end, mid + 1, right, idx * 2 + 1)
            self.tree[idx] = self.lazy[idx] + max(self.tree[idx * 2], self.tree[idx * 2 + 1])

    def book(self, start: int, end: int) -> int:
        self.update(start, end - 1, 0, 10 ** 9, 1)
        return self.tree[1]


def minEatingSpeed(piles: List[int], h: int):
    left, right = 1, max(piles)
    while left < right:
        mid = left + ((right - left) >> 1)
        total = sum([math.ceil(x / mid) for x in piles])
        if total >= h:
            left = mid
        else:
            right = mid - 1
    return left


def strongPasswordCheckerII(password: str) -> bool:
    temp = '!@#$%^&*()-+'
    n = len(password)
    if n < 8:
        print('n < 8')
        return False
    flag1, flag2, flag3, flag4 = 0, 0, 0, 0
    for i in range(n - 1):
        if password[i] == password[i + 1]:
            print(f'{password[i]} = {password[i + 1]}')
            return False
        if password[i] in temp:
            flag4 = 1
        elif password[i].isdigit():
            flag3 = 1
        elif password[i].islower():
            flag1 = 1
        elif password[i].isupper():
            flag2 = 1
    print(f'{flag1} / {flag2} / {flag3} / {flag4}')
    return True if (flag1 and flag2 and flag3 and flag4) else False


def minPathCost(grid: List[List[int]], moveCost: List[List[int]]) -> int:
    m, n = len(grid), len(grid[0])
    ans = [[0 for _ in range(n)] for _ in range(m)]
    path_sum = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            path_sum[i][j] = grid[i][j]
    for i in range(1, m):
        for j in range(n):
            idx, num = -1, 20000
            for k in range(n):
                index = path_sum[i - 1][k]
                if (ans[i - 1][k] + moveCost[grid[i - 1][k]][j]) < num:
                    num = ans[i - 1][k] + moveCost[grid[i - 1][k]][j]
                    idx = index
            ans[i][j] = num
            path_sum[i][j] = grid[i][j] + idx
    index = 0
    maxSum = math.inf
    for j in range(n):
        if ans[m - 1][j] < maxSum:
            maxSum = ans[m - 1][j]
            index = j
    return path_sum[m - 1][index] + ans[m - 1][index]


x, y = defaultdict(int), defaultdict(int)


def work(n: int):
    for i in range(2, int(math.sqrt(n))):
        if n % i == 0:
            cur = 0
            while n % i == 0:
                n //= i
                cur += 1
            if (cur % 2):
                x[i] += 1
            else:
                y[i] += 1
    if n > 1:
        x[n] += 1

def solute(n, temp: List[int]):
    ans = 0
    for v in temp:
        work(v)
    for i in range(100005):
        if (x[i] != 0 and x[i] < n):
            ans += min(x[i], n - x[i])
        elif (x[i] != 0 and y[i] != 0):
            ans += min(x[i], y[i])
    return ans


if __name__ == "__main__":
    n = 3
    temp = [2,1,2]
    print(solute(n, temp))
    # a = "dig1 9 8 6 23"
    # print(a.split(" ", 1))
    # print(1 << 0)
    # print(canIWin(10, 11))
    # a = "IloveLe3tcode!"
    # print(strongPasswordCheckerII(a))
    # a = [[5, 3], [4, 0], [2, 1]]
    # b = [[9, 8], [1, 5], [10, 12], [18, 6], [2, 4], [14, 3]]
    # print(minPathCost(a, b))
    # a = [1,2,3,4,5]
    # print(sum(a[:0]))
    # print(trailingZeroes(5))
    # knightProbability(3, 2, 0, 0)
    # a = eval([64, 64, 64])
    # print(len(a))
    # a = [0, 1, 0, 1, 1, 0, 0]
    # b = [0, 1, 1, 1, 0, 0, 1, 1, 0]
    # c = [1, 1, 0, 0, 1]
    # print(minSwaps(a))
    # a = [1, 2, 3, 4]
    # print(dominantIndex(a))
    # a = [2,2,3]
    # print(permuteUnique(a))
    # a = "B0B6G0R6R0R6G9"
    # b = "B0R0G0R9R0B0G0"
    # c = "G4"
    # print(countPoints(a))
    # arr = [1, 2, 3, 5, 8, 10]
    # print(kIncreasing(arr, 2))
    # p = [[0, 9], [4, 1], [5, 7], [6, 2], [7, 4], [10, 9]]
    # q = [1, 9]
    #
    # a = [3, 2, 1, 4]
    # b = [8, 6, 7, 7]
    # c = [1]
    # price = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 4, 3, 10, 9, 8, 7]
    # print(getDescentPeriods(price))
    # print((getDescentPeriods(a)))
    # print(getDescentPeriods(b))
    # print(p + [q])
    # print(maxTotalFruits(p, 5, 4))
    # print(bisect.bisect_left(arr, 11))
    # k = 2
    # # print(superPow(k, arr))
    # print(maxSumOfThreeSubarrays(arr, 2))
    # print(kthSmallestPrimeFraction(arr, k))
    # a = [[230, 22, 462], [135, 15, 201], [281, 93, 362], [238, 202, 140], [19, 172, 224], [49, 162, 270],
    #      [42, 281, 324], [178, 145, 94], [262, 237, 314], [127, 267, 317], [193, 242, 318], [17, 83, 21], [135, 33, 34],
    #      [86, 39, 272], [183, 37, 339], [14, 114, 492], [204, 283, 191], [289, 260, 162], [209, 144, 214],
    #      [13, 48, 159], [172, 219, 488], [184, 283, 392], [175, 212, 162], [292, 39, 71], [186, 176, 315],
    #      [21, 33, 435], [78, 163, 316], [242, 57, 340], [130, 39, 386], [239, 54, 260], [5, 177, 37], [88, 264, 484],
    #      [49, 46, 409], [230, 195, 164], [259, 150, 465], [279, 268, 29], [53, 150, 222], [116, 47, 210],
    #      [284, 79, 176], [34, 253, 22], [205, 16, 413], [47, 230, 26], [85, 34, 130], [26, 245, 431], [76, 104, 420],
    #      [153, 267, 217], [106, 250, 161], [114, 111, 139], [18, 139, 215], [174, 225, 219], [183, 79, 49],
    #      [220, 23, 489], [63, 177, 48], [210, 2, 167], [230, 243, 211], [132, 78, 196], [148, 259, 258], [81, 177, 161],
    #      [97, 176, 316], [231, 108, 378], [36, 205, 384], [291, 55, 279], [235, 94, 479], [100, 92, 261],
    #      [129, 230, 260], [191, 127, 320], [77, 114, 402], [186, 135, 134], [256, 21, 8], [208, 291, 477],
    #      [117, 188, 2], [32, 95, 67], [265, 225, 19], [254, 51, 294], [51, 126, 288], [285, 73, 497], [282, 190, 87],
    #      [53, 208, 187], [288, 184, 359], [249, 217, 259], [235, 82, 202], [202, 171, 408], [192, 88, 336],
    #      [48, 151, 80], [221, 106, 217], [142, 243, 459], [68, 70, 47], [66, 259, 135], [31, 132, 171], [246, 249, 308],
    #      [122, 35, 49], [270, 263, 134], [290, 210, 319], [232, 70, 364], [91, 28, 279], [159, 47, 362], [32, 124, 487],
    #      [101, 110, 426], [110, 21, 113], [291, 150, 457], [5, 175, 474], [165, 175, 168], [264, 71, 392],
    #      [194, 168, 348], [190, 244, 458], [169, 75, 94], [211, 225, 220], [226, 222, 98], [272, 247, 476],
    #      [89, 127, 312], [90, 172, 316], [214, 192, 339], [141, 187, 489], [126, 16, 452], [258, 20, 465],
    #      [254, 264, 303], [127, 257, 342], [106, 286, 45], [35, 141, 359], [107, 225, 147], [219, 243, 315],
    #      [216, 277, 460], [201, 52, 102], [29, 180, 153], [193, 109, 398], [31, 146, 352], [240, 16, 449],
    #      [171, 129, 318], [290, 65, 283], [83, 149, 258], [228, 166, 475], [84, 88, 409], [171, 224, 209],
    #      [266, 48, 205], [71, 91, 64], [143, 55, 37], [254, 274, 444], [165, 26, 159], [249, 175, 175], [225, 84, 150],
    #      [181, 241, 114], [206, 46, 498], [9, 166, 462], [260, 245, 56], [213, 255, 27], [238, 225, 29],
    #      [100, 212, 164], [6, 226, 59], [222, 218, 155], [78, 68, 325], [216, 228, 453], [197, 274, 440],
    #      [259, 217, 136], [93, 122, 480], [259, 22, 479], [245, 284, 422], [23, 53, 212], [98, 187, 323], [25, 18, 239],
    #      [271, 222, 133], [237, 86, 370], [185, 97, 1], [165, 267, 41], [273, 267, 252], [193, 147, 127],
    #      [205, 48, 174], [91, 193, 340], [10, 63, 368], [50, 126, 293], [122, 7, 363], [121, 55, 305], [84, 69, 350],
    #      [186, 20, 260], [42, 269, 495], [70, 4, 312], [201, 168, 238], [14, 114, 9], [141, 29, 187], [106, 129, 161],
    #      [21, 61, 338], [21, 261, 109], [290, 284, 268], [240, 173, 447], [201, 215, 226], [249, 199, 185],
    #      [163, 253, 85], [1, 224, 254], [90, 260, 182], [217, 138, 44], [1, 33, 75], [110, 94, 134], [167, 252, 303],
    #      [147, 103, 367], [208, 126, 287], [122, 240, 213], [117, 42, 370], [137, 59, 188], [106, 232, 435],
    #      [235, 126, 7], [229, 54, 359], [288, 199, 337], [76, 74, 133], [172, 24, 323], [206, 211, 54], [192, 257, 7],
    #      [84, 210, 314], [242, 207, 433], [271, 20, 5], [247, 39, 434], [49, 120, 296], [170, 33, 21], [87, 21, 152],
    #      [198, 107, 243], [139, 231, 182], [39, 149, 323], [262, 130, 327], [117, 40, 395], [277, 126, 149],
    #      [28, 157, 251], [184, 177, 357], [182, 113, 256], [293, 115, 271], [274, 222, 90], [53, 234, 218],
    #      [69, 289, 173], [221, 116, 253], [215, 27, 80], [20, 165, 345], [128, 85, 428], [255, 149, 194],
    #      [218, 215, 339], [174, 186, 452], [24, 288, 377], [50, 25, 243], [244, 250, 449], [266, 95, 150],
    #      [216, 57, 160], [293, 239, 235], [179, 119, 409], [226, 196, 263], [63, 70, 375], [259, 97, 303],
    #      [35, 156, 257], [271, 269, 439], [181, 6, 353], [260, 201, 5], [207, 228, 96], [183, 234, 433], [250, 178, 25],
    #      [171, 41, 318], [92, 202, 444], [111, 79, 135], [174, 233, 20], [54, 30, 433], [106, 287, 450],
    #      [158, 271, 398], [211, 261, 267], [37, 39, 71], [144, 173, 135], [186, 130, 396], [252, 279, 28],
    #      [226, 265, 324], [227, 161, 198], [213, 140, 155], [135, 231, 159], [21, 94, 398], [27, 11, 162], [48, 9, 271],
    #      [114, 209, 149], [128, 218, 418], [44, 6, 285], [73, 210, 65], [106, 283, 109], [264, 293, 59], [204, 21, 482],
    #      [153, 168, 9], [279, 48, 64], [200, 139, 300], [87, 132, 436], [93, 15, 354], [159, 187, 198], [189, 55, 462],
    #      [284, 17, 461], [199, 185, 128], [240, 289, 235], [189, 181, 219], [207, 85, 353], [62, 33, 211],
    #      [115, 123, 145], [178, 157, 352], [174, 29, 48], [281, 141, 73], [245, 219, 398], [114, 143, 208],
    #      [132, 119, 70], [257, 3, 457], [40, 86, 64], [39, 151, 147], [117, 270, 217], [168, 230, 451], [26, 9, 286],
    #      [241, 233, 372], [96, 221, 112], [8, 212, 318], [37, 38, 137], [193, 276, 391], [157, 85, 200],
    #      [225, 212, 143], [105, 288, 263], [274, 257, 463], [182, 234, 95], [24, 13, 70], [197, 243, 293],
    #      [37, 112, 490], [198, 116, 98], [129, 221, 281], [289, 266, 309], [277, 127, 355], [241, 260, 42],
    #      [142, 181, 240], [116, 3, 150], [87, 159, 439], [253, 123, 214], [157, 216, 143], [283, 200, 75],
    #      [195, 235, 34], [241, 121, 128], [205, 30, 407], [142, 18, 322], [45, 104, 339], [4, 230, 156], [63, 161, 119],
    #      [147, 77, 219], [62, 160, 328], [204, 182, 213], [33, 149, 403], [26, 134, 235], [151, 252, 440],
    #      [77, 193, 342], [149, 175, 500], [204, 212, 317], [248, 23, 218], [92, 87, 217], [52, 62, 289], [80, 45, 118],
    #      [23, 81, 343], [139, 102, 436], [208, 158, 119], [174, 182, 385], [80, 202, 113], [170, 65, 404],
    #      [284, 18, 106], [107, 152, 308], [101, 292, 77], [219, 203, 129], [157, 260, 238], [269, 52, 323],
    #      [187, 207, 380], [148, 90, 136], [182, 80, 478], [28, 126, 349], [50, 157, 242], [68, 44, 414],
    #      [234, 181, 148], [266, 56, 473], [214, 215, 270], [190, 91, 77], [193, 112, 77], [60, 211, 60], [108, 82, 209],
    #      [71, 156, 382], [93, 246, 273], [156, 79, 344], [111, 189, 174], [147, 133, 123], [23, 176, 206],
    #      [242, 48, 227], [175, 24, 351], [192, 255, 470], [259, 98, 264], [14, 17, 133], [276, 278, 268],
    #      [149, 213, 393], [58, 109, 354], [143, 175, 468], [110, 77, 311], [145, 228, 41], [143, 192, 159],
    #      [67, 269, 289], [162, 248, 180], [94, 77, 312], [59, 111, 44], [172, 243, 110], [260, 15, 471], [229, 13, 135],
    #      [20, 76, 98], [261, 195, 60], [187, 12, 412], [268, 1, 487], [249, 150, 191], [15, 101, 297], [89, 182, 152],
    #      [27, 285, 428], [167, 119, 282], [63, 273, 352], [204, 115, 248], [121, 188, 214], [100, 214, 41],
    #      [81, 98, 185], [285, 159, 371], [171, 174, 50], [114, 202, 361], [274, 89, 38], [243, 108, 166],
    #      [276, 58, 244], [237, 91, 437], [171, 53, 157], [105, 157, 45], [224, 90, 480], [232, 266, 187],
    #      [89, 269, 370], [192, 22, 122], [227, 117, 46], [14, 283, 349], [20, 134, 404], [8, 92, 64], [31, 87, 81],
    #      [257, 131, 11], [166, 9, 234], [214, 276, 28], [151, 105, 117], [133, 208, 186], [206, 7, 155], [252, 60, 488],
    #      [11, 210, 500], [290, 179, 367], [282, 59, 219], [136, 35, 171], [221, 275, 371], [70, 197, 218],
    #      [293, 31, 444], [159, 271, 435], [76, 3, 462], [73, 19, 156], [224, 149, 119], [109, 156, 204], [77, 152, 101],
    #      [68, 205, 342], [271, 100, 231], [50, 45, 86], [130, 183, 54], [264, 205, 469], [0, 37, 82], [131, 14, 487],
    #      [260, 115, 220], [45, 111, 362], [203, 57, 164], [228, 44, 499], [252, 22, 441], [61, 51, 209],
    #      [245, 166, 413], [92, 221, 168], [61, 283, 214], [207, 99, 118], [96, 174, 230], [206, 186, 128],
    #      [151, 191, 115], [216, 2, 249], [257, 130, 256], [189, 24, 406], [155, 210, 381], [283, 199, 96],
    #      [260, 147, 213], [239, 242, 44], [283, 7, 240], [47, 248, 372], [26, 253, 383], [126, 194, 195], [83, 161, 31],
    #      [249, 232, 384], [173, 113, 231], [40, 46, 185], [235, 237, 461], [137, 18, 291], [25, 52, 263],
    #      [227, 183, 131], [212, 254, 222], [97, 45, 213], [241, 168, 237], [188, 61, 231], [203, 9, 134],
    #      [17, 144, 123], [259, 100, 157], [183, 264, 248], [48, 247, 168], [256, 99, 322], [97, 47, 175],
    #      [89, 107, 122], [207, 130, 450], [192, 102, 370], [211, 244, 318], [143, 199, 117], [257, 136, 115],
    #      [273, 196, 370], [254, 230, 445], [263, 293, 63], [38, 124, 486], [147, 152, 338], [258, 274, 417],
    #      [242, 276, 104], [271, 98, 158], [223, 188, 30], [62, 86, 206], [178, 191, 412], [13, 16, 490],
    #      [292, 261, 107], [74, 47, 474], [56, 250, 401], [19, 234, 406], [8, 200, 328], [147, 118, 12], [224, 252, 351],
    #      [64, 155, 18], [159, 283, 326], [51, 115, 273], [215, 285, 204], [88, 219, 156], [258, 233, 487],
    #      [101, 16, 85], [116, 228, 480], [115, 234, 10], [228, 140, 405], [16, 131, 285], [65, 10, 491], [13, 73, 202],
    #      [261, 228, 32], [127, 98, 494], [200, 225, 68], [131, 212, 430], [246, 165, 496], [26, 144, 354],
    #      [42, 178, 162], [90, 229, 260], [277, 74, 21], [240, 221, 259], [117, 211, 66], [26, 93, 277], [235, 81, 138],
    #      [130, 209, 14], [73, 131, 66], [184, 69, 455], [160, 77, 10], [188, 293, 223], [89, 217, 130], [88, 10, 158],
    #      [4, 34, 66], [201, 192, 441], [209, 242, 134], [129, 83, 267], [229, 78, 451], [109, 191, 358], [36, 218, 96],
    #      [187, 150, 356], [103, 222, 427], [204, 165, 230], [102, 59, 401], [243, 57, 411], [289, 213, 362],
    #      [74, 206, 358], [239, 202, 443], [193, 218, 410], [192, 242, 154], [46, 196, 294], [37, 71, 168],
    #      [160, 240, 347], [130, 293, 364], [56, 233, 133], [104, 89, 185], [257, 198, 330], [187, 111, 466],
    #      [14, 196, 367], [70, 201, 262], [63, 166, 158], [109, 178, 228], [228, 210, 350], [54, 260, 158],
    #      [51, 257, 34], [5, 94, 487], [57, 287, 274], [194, 79, 72], [123, 87, 170], [227, 144, 484], [228, 211, 30],
    #      [109, 217, 297], [129, 195, 223], [87, 165, 368], [124, 261, 4], [65, 31, 125], [54, 162, 81], [168, 280, 24],
    #      [142, 93, 13], [263, 210, 236], [49, 84, 446], [269, 137, 377], [7, 59, 137], [292, 54, 16], [227, 192, 286],
    #      [13, 32, 496], [148, 177, 391], [250, 273, 231], [281, 40, 483], [164, 91, 136], [209, 134, 218],
    #      [108, 55, 178], [6, 38, 134], [10, 160, 211], [118, 247, 423], [273, 84, 297], [46, 192, 334], [4, 222, 460],
    #      [29, 175, 200], [79, 183, 411], [95, 126, 308], [45, 35, 377], [285, 211, 204], [58, 95, 44], [205, 101, 135],
    #      [63, 149, 436], [213, 230, 231], [175, 92, 169], [49, 291, 439], [259, 284, 46], [175, 170, 316],
    #      [16, 208, 263], [77, 98, 69], [247, 190, 260], [100, 79, 152], [153, 65, 117], [267, 76, 474], [35, 8, 488],
    #      [187, 147, 315], [126, 181, 46], [290, 119, 279], [5, 48, 1], [164, 138, 317], [275, 223, 161], [198, 49, 448],
    #      [194, 165, 319], [196, 178, 217], [254, 47, 479], [293, 260, 267], [276, 277, 207], [266, 48, 84],
    #      [234, 141, 151], [126, 86, 411], [140, 125, 122], [205, 238, 164], [217, 124, 435], [10, 240, 221],
    #      [168, 288, 267], [174, 163, 65]]
    # m = sorted(a, key=(lambda x: [x[2], x[0]]))
    # print(m)
    # b = findAllPeople(294, a, 27)
    # b.sort()
    # c = [0,4,11,13,22,24,27,32,37,38,42,55,59,63,69,70,71,73,79,85,87,88,102,106,112,124,128,139,149,156,159,168,175,183,184,189,190,192,193,194,199,201,205,210,211,214,215,218,222,225,230,232,234,244,249,250,255,259,261,264,269,271,283,285,287,288]
    # print(b)
    # print(c)
    # d = []
    # for x in c:
    #     if x not in b:
    #         d.append(x)
    # print(d)
    # a = [-1, -2, 1, 8, 6, 2, 5, 4, 8, 3, 7]
    # print(maxArea(a))
    # print(threeSum(a))
