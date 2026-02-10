class BSTNode:
    def __init__(self, volatility, symbol):
        self.volatility = volatility
        self.symbol = symbol
        self.left = None
        self.right = None


class VolatilityBST:
    def __init__(self):
        self.root = None

    def insert(self, volatility, symbol):
        if self.root is None:
            self.root = BSTNode(volatility, symbol)
        else:
            self._insert(self.root, volatility, symbol)

    def _insert(self, node, volatility, symbol):
        if volatility < node.volatility:
            if node.left is None:
                node.left = BSTNode(volatility, symbol)
            else:
                self._insert(node.left, volatility, symbol)
        else:
            if node.right is None:
                node.right = BSTNode(volatility, symbol)
            else:
                self._insert(node.right, volatility, symbol)

    def get_priority_list(self):
        """Returns [(symbol, volatility)] from highest → lowest volatility"""
        result = []
        self._reverse_inorder(self.root, result)
        return result

    def _reverse_inorder(self, node, result):
        if node:
            self._reverse_inorder(node.right, result)
            result.append((node.symbol, node.volatility))
            self._reverse_inorder(node.left, result)

    def print_priority_list(self):
        """Pretty print the BST priority order."""
        priority = self.get_priority_list()

        print("\n📊 Volatility Priority (High → Low)")
        print("-----------------------------------")

        for symbol, vol in priority:
            print(f"{symbol}: {vol:.4f}")

        print("-----------------------------------\n")
