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
