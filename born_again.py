import math

from trees import Node, NodeType, TreeFile

class FeatureSpace:
    def __init__(self, random_forest):
        self.random_forest = random_forest
        
    def initialize_cells(self):
        self.ordered_hyperplane_levels = self.random_forest.get_hyperplanes()
        self.nb_features = len(self.ordered_hyperplane_levels)
        # temp values used when recursively enumerating cells
        self.temp_cell_values = [-1 for _ in range(self.nb_features)]
        self.nb_cells = 1
        self.nb_possible_regions = 1.0
        for feature_hyperplanes in self.ordered_hyperplane_levels:
            nb_hyperplanes = len(feature_hyperplanes)
            self.nb_cells *= nb_hyperplanes
            self.nb_possible_regions *= nb_hyperplanes * (nb_hyperplanes + 1.0) / 2.0
            
        self.strides = [1 for _ in range(self.nb_features)]
        for i in reversed(range(self.nb_features-1)):
            self.strides[-i] = self.strides[-i-1] * len(self.ordered_hyperplane_levels[-i-1])
        
        self.cells = [-1 for _ in range(self.nb_cells)]
        self.enumerate_cells(0, 0)
    
    def enumerate_cells(self, k, cell_index):
        if (k == self.nb_features):
            self.cells[cell_index] = self.random_forest.majority_class(self.temp_cell_values)
        else:
            stride_value = self.strides[k]
            for i, level in enumerate(self.ordered_hyperplane_levels[k]):
                self.temp_cell_values[k] = level
                self.enumerate_cells(k+1, cell_index + stride_value * i)
                
    def key_to_hash(self, key_bottom_left, key_top_right):
        key_left = key_bottom_left
        key_right = key_top_right
        code = 1
        newKey = 0
        for k in reversed(range(self.nb_features)):
            size = len(self.ordered_hyperplane_levels[k])
            if size != 1:
                valueLeft = key_left % size
                valueRight = key_right % size
                key_left = key_left // size
                key_right = key_right // size
                newKey += (valueRight - valueLeft)*code
                code *= size - valueLeft
        return newKey
    
    def key_to_cell(self, key, k):
        if k > 0:
            return (key % self.strides[k-1]) // self.strides[k]
        else:
            return key // self.strides[k]

class BornAgainTree:
    def __init__(self, random_forest):
        self.random_forest = random_forest
        self.reborn_tree = []
        
    def build_optimal(self):
        self.fspace = FeatureSpace(self.random_forest)
        self.fspace.initialize_cells()
        self.regions = [
            [math.inf] * (self.fspace.key_to_hash(i, self.fspace.nb_cells-1) + 1)
            for i in range(self.fspace.nb_cells)
        ]
        final_objective = self.optimize_depth(0, self.fspace.nb_cells - 1)
        self.collect_result(0, self.fspace.nb_cells - 1, final_objective, 0)
        
    def optimize_depth(self, index_bottom, index_top):
        if index_bottom == index_top: return 0
        
        hash_value = self.fspace.key_to_hash(index_bottom, index_top)
        if (self.regions[index_bottom][hash_value] != math.inf):
            return self.regions[index_bottom][hash_value]
        
        best_lb = 0
        best_ub = math.inf
        
        for feature in range(self.random_forest.nb_features):
            if best_lb >= best_ub: break
            
            stride = self.fspace.strides[feature]
            range_low = self.fspace.key_to_cell(index_bottom, feature)
            range_up = self.fspace.key_to_cell(index_top, feature)
            temp_range_low = range_low
            temp_range_up = range_up
            while temp_range_low < temp_range_up and best_lb < best_ub:
                l = (temp_range_low + temp_range_up) // 2
                left_result = self.optimize_depth(index_bottom, index_top + stride * (l - range_up))
                if left_result > best_lb: best_lb = left_result
                right_result = self.optimize_depth(index_bottom + stride * (l + 1 - range_low), index_top)
                if left_result == 0 and right_result == 0:
                    if self.fspace.cells[index_bottom] == self.fspace.cells[index_top]:
                        self.regions[index_bottom][hash_value] = 0
                        return 0
                    else:
                        self.regions[index_bottom][hash_value] = 1
                        return 1
                if right_result > best_lb: best_lb = right_result
                if (1 + right_result < best_ub): best_ub = 1 + max(left_result, right_result)
                if (1 + left_result  >= best_ub): temp_range_up = l
                if (1 + right_result >= best_ub): temp_range_low = l + 1
        
        self.regions[index_bottom][hash_value] = best_ub
        return best_ub
    
    def collect_result(self, index_bottom, index_top, opt_value, current_depth):
        if opt_value == 0:
            node = Node(
                node_id=len(self.reborn_tree),
                node_type=NodeType.Leaf,
                split_value=-1,
                depth=current_depth,
                classification=self.fspace.cells[index_bottom]
            )
            self.reborn_tree.append(node)
            return node.node_id
        else:
            for feature in range(self.random_forest.nb_features):
                stride = self.fspace.strides[feature]
                range_low = self.fspace.key_to_cell(index_bottom, feature)
                range_up = self.fspace.key_to_cell(index_top, feature)
                for l in range(range_low, range_up):
                    index_top_left = index_top + stride * (l - range_up)
                    hash1 = self.fspace.key_to_hash(index_bottom, index_top_left)
                    if (index_bottom == index_top_left): 
                        left_result = 0
                    else:
                        left_result = self.regions[index_bottom][hash1]

                    index_bottom_right = index_bottom + stride * (l + 1 - range_low)
                    hash2 = self.fspace.key_to_hash(index_bottom_right, index_top)
                    if (index_bottom_right == index_top):
                        right_result = 0
                    else:
                        right_result = self.regions[index_bottom_right][hash2]

                    # found the optimal split used by the DP algorithm
                    if left_result != math.inf and right_result != math.inf and (1 + max(left_result, right_result) == opt_value):
                        node_id = len(self.reborn_tree)
                        self.reborn_tree.append(Node())
                        self.reborn_tree[node_id].node_id = node_id
                        self.reborn_tree[node_id].node_type = NodeType.Internal
                        self.reborn_tree[node_id].split_feature = feature
                        self.reborn_tree[node_id].split_value = self.fspace.ordered_hyperplane_levels[feature][l]
                        left_id =  self.collect_result(index_bottom, index_top_left, left_result, current_depth + 1)
                        right_id = self.collect_result(index_bottom_right, index_top, right_result, current_depth + 1)
                        self.reborn_tree[node_id].left_child = left_id
                        self.reborn_tree[node_id].right_child = right_id
                        self.reborn_tree[node_id].depth = current_depth
                        return node_id
                    
    def export(self, filename, print_content=False):
        TreeFile.export(filename, "", "BA", self.random_forest.nb_features, self.random_forest.nb_classes, 
                        [self.reborn_tree], print_content=print_content)