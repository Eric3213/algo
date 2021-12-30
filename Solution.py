from typing import List
from Trie import *
from test_tree import *
from test_array import *


class Solution:

    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        words.sort(key=len)
        ans = []
        root = Trie()
        for word in words:
            if word == "":
                continue
            if root.dfs(word, 0):
                ans.append(word)
            else:
                root.insert(word)
        return ans

    def zigzagLevelOrder(self, tree: TreeNode):
        t = Solution()
        return t.zigzagLevelOrder(tree)


