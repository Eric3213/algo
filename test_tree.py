# Definition for a binary tree node.
from collections import deque
from math import inf
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def print_1(self, r):
        if not r:
            return
        self.print_1(r.left)
        self.print_1(r.right)
        print(r.val)
        return


class Solution:
    def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
        def lowestCommonAncestor(root: TreeNode, p: int, q: int) -> TreeNode:
            if not root or root.val == p or root.val == q:
                return root
            left = lowestCommonAncestor(root.left, p, q)
            right = lowestCommonAncestor(root.right, p, q)
            if not left: return right
            if not right: return left
            return root

        def findPath(root: TreeNode, s: int, e: int):
            if not root:
                return

            ans1.append("L")
            print(ans1)
            findPath(root.left, s, e)
            ans1.pop()
            print(ans1)
            ans1.append("R")
            print(ans1)
            findPath(root.right, s, e)
            ans1.pop()
            print(ans1)
            if root.val == s:
                res1.append(ans1[:])
                print("res1: " + str(res1))
                n = len(ans1)
                print(n)
            if root.val == e:
                res2.append(ans1[:])
                print("res2: " + str(res2))

        root = lowestCommonAncestor(root, startValue, destValue)
        # print(root.val)

        ans1 = []
        res1 = []
        res2 = []
        n = 0

        findPath(root, startValue, destValue)
        print(str(res1))
        print(str(res2))
        ans1 = ["U"] * n
        print(str(ans1))
        return ''.join(ans1 + res2[0])

    def zigzagLevelOrder(self, root: TreeNode):
        if not root: return []
        ans = []
        i = 0
        q = [root]
        while q:
            i += 1
            if i % 2:
                ans.append([node.val for node in q])
            else:
                ans.append([node.val for node in q[::-1]])
            temp = []
            for node in q:
                if node.left:
                    temp.append(node.left)
                if node.right:
                    temp.append(node.right)
            q = temp
        return ans

    def maxPathSum(self, root: TreeNode):
        self.maxSum = float('-inf')

        def dfs(root: TreeNode):
            if not root:
                return 0
            left = max(dfs(root.left), 0)
            right = max(dfs(root.right), 0)
            self.maxSum = max(self.maxSum, root.val + left + right)
            return root.val + max(left, right)

        dfs(root)
        return self.maxSum

    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.maxSum = float('-inf')

        def dfs(root):
            if not root:
                return 0
            left = max(dfs(root.left), 0)
            right = max(dfs(root.right), 0)
            self.maxSum = max(self.maxSum, 1 + left + right)
            return max(left, right) + 1

        dfs(root)
        return self.maxSum

    def isBalanced(self, root: TreeNode) -> bool:

        def dfs(root):
            if not root:
                return 0
            leftH = dfs(root.left)
            rightH = dfs(root.right)
            if leftH == -1 or rightH == -1 or abs(leftH - rightH) > 1:
                return -1
            return max(leftH, rightH) + 1

        return dfs(root) >= 0

    def successor(self, root: TreeNode):
        root = root.right
        while root.left:
            root = root.left
        return root.val

    def predecessor(self, root: TreeNode):
        root = root.left
        while root.right:
            root = root.right
        return root.val

    # 删除二叉搜索树的Node
    def deleteNode(self, root: TreeNode, key: int) -> TreeNode:
        if not root:
            return None
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        else:
            if not root.left and not root.right:
                return None
            elif root.right:
                root.val = self.successor(root)
                root.right = self.deleteNode(root.right, root.val)
            else:
                root.val = self.predecessor(root)
                root.left = self.deleteNode(root.left, root.val)
        return root

    def isValidBST(self, root: TreeNode):
        # def helper(node, lower=float('-inf'), upper=float('inf')):
        #     if not node:
        #         return True
        #     val = node.val
        #     if val <= lower or val >= upper:
        #         return False
        #     if not helper(node.left, lower, val):
        #         return False
        #     if not helper(node.right, val, upper):
        #         return False
        #     return True
        #
        # return helper(root)
        stack, inorder = [], float('-inf')
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if root.val <= inorder:
                return False
            inorder = root.val
            root = root.right
        return True

    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode):
        """
        递归
        :param self:
        :param root:
        :param p:
        :param q:
        :return: 最近公共祖先
        """
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left and not right:
            return
        if not left:
            return right
        if not right:
            return left
        return root



class Codec:

    # def serialize(self, root):
    #     if not root:
    #         return "None"
    #     return str(root.val) + ',' + self.serialize(root.left) + ',' + self.serialize(root.right)
    #
    # def deserialize(self, data):
    #
    #     def dfs(datalist):
    #         val = datalist.pop(0)
    #         if val == 'None':
    #             return None
    #         root = TreeNode(int(val))
    #         root.left = dfs(datalist)
    #         root.right = dfs(datalist)
    #         return root
    #
    #     datalist = data.split(',')
    #     return dfs(datalist)
    def serialize(self, root: TreeNode):
        ans = []
        def postOrder(root: TreeNode):
            if root is None:
                return
            postOrder(root.left)
            postOrder(root.right)
            ans.append(root.val)
        postOrder(root)
        return " ".join(map(str, ans))

    def deserialize(self, data):
        ans = list(map(int, data.split()))
        def construct(lower, upper):
            if ans == [] or ans[-1] < lower or ans[-1] > upper:
                return None
            val = ans.pop()
            root = TreeNode(val)
            root.right = construct(val, upper)
            root.left = construct(lower, val)
            return root
        root = construct(-inf, inf)
        return root



def inorderTraversal(self, root: TreeNode):
    """
    法一：递归
    法二：迭代
    法三：莫里斯
    x 没有左节点：
        x 加入答案  x = x.right
    x 有左节点：
        x 的predecessor 右孩子为空： 右孩子指向 x； x = x.left
        x 的predecessor 右孩子不为空： x 加入答案； x = x.right
    :param self:
    :param root:
    :return:
    """
    # def dfs(cur):
    #     if not cur:
    #         return
    #
    #     dfs(cur.left)
    #     ans.append(cur.val)
    #     dfs(cur.right)
    #
    # ans = []
    # dfs(root)
    # return ans

    # 法2
    # ans = []
    # stack = []
    # cur = root
    # while stack or cur:
    #     while cur:
    #         stack.append(cur)
    #         cur = cur.left
    #     cur = stack.pop()
    #     ans.append(cur.val)
    #     cur = cur.right
    # return ans

    # 法3
    ans = []
    cur = root
    while cur:
        if not cur.left:
            ans.append(cur.val)
            cur = cur.right
        else:
            pre = cur.left
            while pre.right and pre.right != cur:
                pre = pre.right
            if not pre.right:
                pre.right = cur
                cur = cur.left
            else:
                ans.append(cur.val)
                cur = cur.right
    return ans


def pre_order(root, ans=None):
    if not root:
        return
    ans.append(root.val)
    pre_order(root.left)
    pre_order(root.right)


class Node:
    def __init__(self, val):
        self.val = val
        self.children = None


def preorder(root: Node):
    if not root:
        return []
    stack = [root]
    ans = list()
    while stack:
        node = stack.pop()
        ans.append(node.val)
        stack.extend(node.children[::-1])
    return ans


def averageOfLevels(root: TreeNode):
    ans = []
    q = deque([root])
    while q:
        total = 0
        size = len(q)
        for _ in range(size):
            node = q.popleft()
            total += node.val
            left, right = node.left, node.right
            if left:
                q.append(left)
            if right:
                q.append(right)
        ans.append(total / size)
    return ans

# ser = Codec()
# deser = Codec()
# root = TreeNode(1)
# left = TreeNode(2)
# right = TreeNode(3)
# root.left = left
# root.right = right
# right.left = TreeNode(4)
# right.right = TreeNode(5)
#
# print(ser.serialize(root))
# ans = []
# pre_order((deser.deserialize(ser.serialize(root))))
# print(ans)

ser = Codec()
root = TreeNode(2)
root.left = TreeNode(1)
root.right = TreeNode(3)
ser.deserialize("1 3 2").print_1(ser.deserialize("1 3 2"))