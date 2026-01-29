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

    # Get stocks from highest → lowest volatility
    def get_priority_list(self):
        result = []
        self._reverse_inorder(self.root, result)
        return result

    def _reverse_inorder(self, node, result):
        if node:
            self._reverse_inorder(node.right, result)
            result.append((node.symbol, node.volatility))
            self._reverse_inorder(node.left, result)
