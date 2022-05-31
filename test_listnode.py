from typing import Optional, List


# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


def removeElements(head: ListNode, val: int):
    """
    :param head:
    :param val:
    :return: 删除所有值为val的node
    """
    dummy = ListNode(next=head)
    cur = dummy
    while cur.next:
        if cur.next.val == val:
            cur.next = cur.next.next
        else:
            cur = cur.next
    return dummy.next


def hasCycle(head: Optional[ListNode]):
    seen = set()
    while head:
        if head in seen:
            return True
        seen.add(head)
        head = head.next
    return False


def hasCycle2(head: Optional[ListNode]):
    fast = slow = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if fast == slow:
            return True
    return False


def deleteDuplicates(head: ListNode):
    """

    :param head: 升序链表
    :return: 删除所有重复节点
    """
    if not head:
        return head
    dummy = ListNode()
    dummy.next = head
    cur = dummy

    while cur.next and cur.next.next:
        if cur.next.val == cur.next.next.val:
            x = cur.next.val
            while cur.next and cur.next.val == x:
                cur.next = cur.next.next
        else:
            cur = cur.next
    return dummy.next


def platesBetweenCandles(s: str, queries: List[List[int]]):
    n = len(s)
    preSum, left, right = [0] * n, [0] * n, [0] * n
    l, r, temp_sum = -1, -1, 0
    for i, ch in enumerate(s):
        if ch == "*":
            temp_sum += 1
        else:
            l = i
        left[i] = l
        preSum[i] = temp_sum
    for i in range(n-1, -1, -1):
        if s[i] == "|":
            r = i
        right[i] = r

    ans = [0] * len(queries)
    for i, (x, y) in enumerate(queries):
        x, y = right[x], left[y]
        if 0 <= x < y and y >= 0:
            ans[i] = preSum[y] - preSum[x]
    return ans


def reverseList(self, head: ListNode):
    """
    反转链表 迭代+递归
    :param head:
    :return:
    """
    # prev, cur = None, head
    # while cur is not None:
    #     next = cur.next
    #     cur.next = prev
    #     prev = cur
    #     cur = next
    # return prev
    if head is None or head.next is None:
        return head
    node = self.reverseList(head.next)
    head.next.next = head
    head.next = None
    return node


class Node:

    def __init__(self, val):
        self.val = val
        self.next = None

class MyLinkedList:

    def __init__(self):
        self._head = Node(0)
        self._count = 0

    def get(self, index: int):
        if 0 <= index < self._count:
            node = self._head
            for _ in range(index+1):
                node = node.next
            return node.val
        else:
            return -1

    def addAtHead(self, val: int):
        return self.addAtIndex(0, val)

    def addAtTail(self, val: int):
        return self.addAtIndex(self._count, val)

    def addAtIndex(self, index: int, val: int):
        if index < 0:
            index = 0
        elif index > self._count:
            return
        # 计数+1
        self._count += 1
        add_node = Node(val)
        prev, cur = None, self._head
        for _ in range(index + 1):
            prev, cur = cur, cur.next
        else:
            prev.next, add_node.next = add_node, cur


    def deleteAtIndex(self, index: int):
        if 0 <= index < self._count:
            self._count -= 1
            prev, cur = None, self._head
            for _ in range(index+1):
                prev, cur = cur, cur.next
            else:
                prev.next, cur.next = cur.next, None


