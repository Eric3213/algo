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
                print("res1: "+str(res1))
                n = len(ans1)
                print(n)
            if root.val == e:
                res2.append(ans1[:])
                print("res2: "+str(res2))

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


if __name__ == "__main__":
    root1 = TreeNode(3, None, None)
    root2 = TreeNode(1, root1, None)
    root3 = TreeNode(6, None, None)
    root4 = TreeNode(4, None, None)
    root5 = TreeNode(2, root3, root4)
    root = TreeNode(5, root2, root5)
    # print(root.val)
    print(Solution.getDirections(self=Solution, root=root, startValue=3, destValue=6))
