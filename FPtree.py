class TreeNode:
    def __init__(self, name_value, num, parent_node):
        self.name = name_value  # 存放节点的名字
        self.count = num  # 计数值
        self.nodeLink = None  # 用于链接相似的元素项
        self.parent = parent_node  # 指向当前节点的父节点
        self.children = {}  # 当前节点的子节点

    def inc(self, num):
        # 对count变量增加给定值
        self.count += num
