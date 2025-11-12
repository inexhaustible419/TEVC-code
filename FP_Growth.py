
from FPtree import TreeNode
import random


class FpGrowth:
    """
    #用于数据挖掘，特别是挖掘频繁项集，主要用于解空间的优化。它的核心思想是通过构建 FP 树 来提高频繁项集的挖掘效率。
    """
    def __init__(self, elites_g_split, min_support, model):
        # self.elites = elites
        self.elites_g_split = elites_g_split
        self.min_support = min_support
        self.model = model
        self.data = []  # 存储所有部分的转换后数据，形如[{[(),(),...]:num,[(),(),...]:num,...},...]
        self.dm_freq_ls = []  # 存储所有部分挖掘后的频繁项集

    def start_cut(self):
        """
        功能:该函数将精英解（elites_g_split）按照地面站进行分组，并返回按地面站划分的解。
        主要步骤：
        遍历 elites_g_split，它是一个包含多个地面站的字典，每个地面站对应多个解。
        将每个地面站的解聚集在一起，存储到 part 字典中，并返回这个字典。
        作用
        通过对解进行地面站级别的划分，为后续的数据挖掘（如频繁项集挖掘）做准备。
        """
        part = {}  # 将解按地面站分离
        # mining_part = {}
        for elite in self.elites_g_split:
            # elite形如{地面站1：[id1,id2,...], 地面站2：[id1,id2,...]}
            for g, sol in elite.items():
                if g not in part.keys():
                    part[g] = [sol]
                else:
                    part[g].append(sol)
        return part

    def choose(self, elites):
        """
        功能:该函数用于从给定的精英解集合中选择不重复的解。

        主要步骤：
        遍历 elites，如果解没有在 elites_choose 列表中，则添加进去。
        作用
        此函数通过去重，确保在数据挖掘时每个解只出现一次，避免重复计算。
        """
        elites_choose = []
        for elite in elites:
            if elite not in elites_choose:
                elites_choose.append(elite)
        return elites_choose

    def transform_data(self, elites):
        """
        功能:将解转换为适合进行频繁项集挖掘的数据格式。
        主要步骤：
        遍历 elites，将每个解中的相邻弧段（(id1, id2)）转换为二元组，表示解的相邻弧段。
        统计每个转换后数据的出现次数，去重，并生成一个 data_dict 字典，记录每个解的出现频率。
        将转换后的数据存储在 self.data 中。
        作用
        通过转换原始解为频繁项集的候选项（相邻弧段对），为后续的 FP 树构建和频繁项集挖掘做准备。
        """
        data_temp = []
        index_temp = []
        for i in range(len(elites)):
            data_item1 = []  # 存储转换后的解，形如[(id1,id2),(id2,id3),(id3,id4),...]
            for j in range(len(elites[i]) - 1):
                index1 = elites[i][j]
                index2 = elites[i][j + 1]
                data_item1.append((index1, index2))  # 两个相邻弧段作为挖掘的对象，即挖掘出的对象为一对对相邻弧段
            data_temp.append(data_item1)  # 存储转换后的所有解
            index_temp.append(1)
        # data_temp得到的是将所有解转换后的数据，需要检查是否有重复数据，若有则统计
        data = []
        index = []
        data.append(data_temp[0])
        index.append(1)
        for i in range(1, len(data_temp)):
            k = 0
            for j in range(len(data)):
                if data_temp[i] == data[j]:
                    index[j] += 1
                    k = 1
                    break
            if k == 0:
                data.append(data_temp[i])
                index.append(1)
        data_dict = {}
        for trans_id in range(len(data)):
            data_dict[frozenset(data[trans_id])] = index[trans_id]  # 字典的键必须是不可变对象
        # 形如{{}:num,...}，{}代表转换后的解且为frozenset形式,num表示该解出现的次数
        self.data.append(data_dict)

    def update_tree(self, items, in_tree, head_table, num):
        """
        功能:用于更新 FP 树，向树中添加新节点，或者增加已有节点的计数。

        主要步骤：
        如果当前项已经在树中，则增加其计数。
        如果当前项不在树中，则新建一个节点并加入树，同时更新 head_table，将新节点链接到相应的位置。
        递归地继续处理后续的项。
        作用
        这一步是构建 FP 树的关键，通过在树中不断添加节点，形成树结构，从而为频繁项集的挖掘奠定基础。
        """
        # items为排序后的数据形如{(),(),...}
        if items[0] in in_tree.children:
            in_tree.children[items[0]].inc(num)
        else:
            # 若不在子结点中则创建节点存储该子节点
            in_tree.children[items[0]] = TreeNode(items[0], num, in_tree)  # TreeNode中参数代表：节点的名字,计数值,父节点
            if head_table[items[0]][1] is None:
                head_table[items[0]][1] = in_tree.children[items[0]]
            else:
                self.update_head(head_table[items[0]][1], in_tree.children[items[0]])
        if len(items) > 1:
            self.update_tree(items[1:], in_tree.children[items[0]], head_table, num)

    @staticmethod
    def update_head(node_to_link, target_node):
        while node_to_link.nodeLink is not None:
            node_to_link = node_to_link.nodeLink
        node_to_link.nodeLink = target_node

    def create_tree(self, data):
        """
        功能:根据给定的数据构建 FP 树，并返回根节点和头表（head_table）。
        主要步骤：
        遍历 data 中的每一条数据，统计各项的支持度。
        去除支持度较小的项。
        按照支持度排序后，构建 FP 树。
        作用
        该函数通过根据支持度生成 FP 树结构，并返回构建好的树和头表，用于后续的频繁项集挖掘。
        """
        head_table = {}
        for trans_elite in data:  # trans_elite = [(id1,id2),...]
            for item in trans_elite:  # item形如(id1,id2)
                head_table[item] = head_table.get(item, 0) + data[trans_elite]  # data[trans_elite]为num
        for item in list(head_table.keys()):
            if head_table[item] < self.min_support:
                del head_table[item]  # head_table形如{(id1,id2):num1, (id3,id4):num2, ...}
        freq_item = set(head_table.keys())  # 删除了支持度小的
        if len(freq_item) == 0:
            return None, None
        for item in head_table:
            head_table[item] = [head_table[item], None]  # head_table形如{(id1,id2):[num1,Node],(id3,id4):[num2,Node] ...}
        root_tree = TreeNode('null set', 1, None)  # 分别代表存放节点的名字,计数值,父节点
        # 根据数据集中的每条数据依次构造fp树
        for trans, num in data.items():  # data形如{{(),(),...}:num,{(),(),...}:num,...}
            local = {}  # 形如{():num,():num...}
            for item in trans:
                if item in freq_item:
                    local[item] = head_table[item][0]  # 一项集
            if len(local) > 0:
                # 对数据按照支持度排序
                sorted_items = [items[0] for items in sorted(local.items(), key=lambda x:x[1], reverse=True)]
                # 对数据构造树节点
                self.update_tree(sorted_items, root_tree, head_table, num)
        return root_tree, head_table

    def ascend_tree(self, leaf_node, pre_path):  # 上溯找到该条条件模式基
        if leaf_node.parent is not None:
            pre_path.append(leaf_node.name)
            self.ascend_tree(leaf_node.parent, pre_path)

    def find_pre_path(self, tree_node): # 为每个树节点构建条件模式基，并计算支持度。
        con_pts = {}  # 存储节点tree_node对应的条件模式基以及对应数量,形如{[条件模式基1]:num1}
        while tree_node is not None:
            pre_path = []
            self.ascend_tree(tree_node, pre_path)
            if len(pre_path) > 1:
                con_pts[frozenset(pre_path[1:])] = tree_node.count
            tree_node = tree_node.nodeLink
        return con_pts

    def mine_tree(self, head_table, pre, freq_item_ls):  # 根据叶子节点的条件模式基得到其对应的所有频繁项集
        '''
        功能:基于 FP 树中的头表挖掘频繁项集。
        主要步骤：
        对头表中的每个项，生成条件模式基并构建条件 FP 树。
        递归地挖掘所有频繁项集，并将结果存储到 freq_item_ls 中。
        作用
        通过对 FP 树进行深度挖掘，找到所有频繁项集，并将它们存储到 freq_item_ls 中。
        '''
        # head_table形如{(id1,id2):[num1,Node],(id3,id4):[num2,Node] ...}
        sort_ls = [v[0] for v in sorted(head_table.items(), key=lambda x:x[1][0])]  # 按照支持度升序排列，形如[(id1, id2)...]
        for base_pt in sort_ls:
            new_freq = pre.copy()
            new_freq.add(base_pt)  # set类型的用add方法添加
            freq_item_ls.append(new_freq)
            # head_table[base_pt][1]表示的是base_pt的node_link, node与node_link的name一样，count不一样
            cond_pt_base = self.find_pre_path(head_table[base_pt][1])
            # 从条件模式基中构建频繁模式树
            cond_tree, cond_head = self.create_tree(cond_pt_base)
            if cond_head is not None:
                self.mine_tree(cond_head, new_freq, freq_item_ls)

    def start_dm(self):
        """
        功能:执行数据挖掘的整个流程，包括数据切分、转换、FP 树构建和频繁项集挖掘。
        主要步骤：
        调用 start_cut() 将解按地面站划分。
        对每个地面站的解，进行数据转换、频繁项集挖掘。
        构建 FP 树并挖掘频繁项集。
        返回所有挖掘出的频繁项集。
        作用
        该函数是 FP-growth 算法的入口，完成了数据挖掘的全过程。
        """
        # 形如{地面站1：[[id1,id2,...],[id1,id3,...],...], 地面站2：[[id1,id2,...],[id1,id3,...],...], ...}
        elite_part = self.start_cut()  # 挖掘过长的解序列会导致爆内存
        i = 0
        for elites_all in elite_part.values():
            # print(i, 'elites_all*********', len(elites_all), elites_all)
            elites = self.choose(elites_all)
            # print(i, 'elites*************', len(elites), elites)
            i += 1
            self.transform_data(elites)
        for data in self.data:  # data形如{[(),(),...]:num,[(),(),...]:num,...}
            (root_tree, head_table) = self.create_tree(data)
            if root_tree is None:
                # self.dm_freq_ls = []
                continue
            freq_item_ls = []
            self.mine_tree(head_table, set(list()), freq_item_ls)
            i_length = [len(i) for i in freq_item_ls]
            index = i_length.index(max(i_length))
            # 找到一个最大频繁项集
            max_freq_items = freq_item_ls[index]
            max_freq_ls = list(max_freq_items)
            chain_list = []  # 找到最大频繁项集中可以按顺序连续组成的所有链
            while len(max_freq_ls) > 0:
                chain = list()
                chain.append(max_freq_ls[0][0])
                chain.append(max_freq_ls[0][1])
                del max_freq_ls[0]
                if len(max_freq_ls) > 0:
                    while True:
                        k = 0
                        for i in range(len(max_freq_ls)):
                            if max_freq_ls[i][1] == chain[0]:
                                chain.insert(0, max_freq_ls[i][0])
                                k = 1
                                del max_freq_ls[i]
                                if len(max_freq_ls) == 0:
                                    k = 0
                                break
                        if k == 0:
                            break
                if len(max_freq_ls) > 0:
                    while True:
                        k = 0
                        for i in range(len(max_freq_ls)):
                            if max_freq_ls[i][0] == chain[len(chain) - 1]:
                                chain.insert(len(chain), max_freq_ls[i][1])
                                k = 1
                                del max_freq_ls[i]
                                if len(max_freq_ls) == 0:
                                    k = 0
                                break
                        if k == 0:
                            break
                chain_list.append(chain)
            if len(chain_list) == 1:
                dm_freq = [arc for arc in chain_list[0]]  # 存储挖掘到的频繁项集
            else:
                lt = []
                for i in range(len(chain_list)):
                    t = 0
                    for arc in chain_list[i]:
                        t += self.model.Arc_list[arc].link_time
                    lt.append(t)
                dm_freq = [arc for arc in chain_list[lt.index(max(lt))]]
            self.dm_freq_ls.append(dm_freq)
        return self.dm_freq_ls

