# Definition for a binary tree node.
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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
        def dfs(root):
            if not root:
                return 0
            left = max(dfs(root.left), 0)
            right = max(dfs(root.right), 0)
            self.maxSum = max(self.maxSum, root.val + left + right)
            return max(left, right) + root.val
        dfs(root)
        return self.maxSum