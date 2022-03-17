class Trie:

    def __init__(self):
        self.children = [None] * 26
        self.isEnd = False

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            ch = ord(ch) - ord("a")
            if not node.children[ch]:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.isEnd = True

    def dfs(self, word: str, start: int) -> bool:
        if start == len(word):
            return True
        node = self
        for i in range(start, len(word)):
            node = node.children[ord(word[i]) - ord("a")]
            if node is None:
                return False
            if node.isEnd and self.dfs(word, i+1):
                return True
        return False


    def searchPrefix(self, prefix: str):
        node = self
        for ch in prefix:
            ch = ord(ch) - ord("a")
            if not node.children[ch]:
                return None
            node = node.children[ch]
        return node

    def search(self, word: str):
        node = self.searchPrefix(word)
        return node is not None and node.isEnd

    def startsWith(self, prefix: str):
        node = self.searchPrefix(prefix)
        return node is not None

# test
# new test


