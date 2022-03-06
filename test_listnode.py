from typing import Optional


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


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
