import re
from enum import Enum

import numpy as np

NodeType = Enum('NodeType', 'Null Internal Leaf')

class Node:
    def __init__(self, node_id=-1, node_type=NodeType.Null, left_child=-1, right_child=-1, 
                 split_feature=-1, split_value=-1e30, depth=-1, classification=-1, average_target=-1):
        self.node_id = node_id
        self.node_type = node_type
        self.left_child = left_child
        self.right_child = right_child
        self.split_feature = split_feature
        self.split_value = split_value
        self.depth = depth
        self.classification = classification
        self.average_target = average_target
        
class RandomForest:
    def __init__(self, filename):
        self.nb_features, self.nb_classes, self.trees = TreeFile.parse(filename)
        
    def majority_class(self, sample):
        votes = [0 for _ in range(self.nb_classes)]
        
        for tree in self.trees:
            node = tree[0]
            while (node.node_type == NodeType.Internal):
                if (sample[node.split_feature] <= node.split_value):
                    node = tree[node.left_child]
                else:
                    node = tree[node.right_child]
            votes[node.classification] += 1
            
        return np.argmax(votes)
        
    def get_hyperplanes(self):
        hyperplanes = [set() for _ in range(self.nb_features)]
        for tree in self.trees:
            for node in tree:
                if node.node_type == NodeType.Internal:
                    hyperplanes[node.split_feature].add(node.split_value)
        return [sorted(h) + [1e30] for h in hyperplanes]

class TreeFile:
    regex_dict = {
        'nb_features': re.compile(r'NB_FEATURES: (?P<nb_features>\d+)'),
        'nb_classes': re.compile(r'NB_CLASSES: (?P<nb_classes>\d+)'),
        'tree_id': re.compile(r'\[TREE (?P<tree_id>\d+)\]'),
        'nb_nodes': re.compile(r'NB_NODES: (?P<nb_nodes>\d+)')
    }
    
    @staticmethod
    def parse_line(line):
            for key, regex in TreeFile.regex_dict.items():
                match = regex.search(line)
                if match:
                    return key, match
            return None, None
    
    @staticmethod
    def parse(filename):
        nb_features = -1
        nb_classes = -1
        trees = []

        with open(filename) as file:
            line = file.readline()
            while line:
                key, match = TreeFile.parse_line(line)

                if key == "nb_features":
                    nb_features = int(match.group(key))

                if key == "nb_classes":
                    nb_classes = int(match.group(key))

                if key == "tree_id":
                    tree_id = int(match.group(key))
                    trees.append([])

                if key == "nb_nodes":
                    nb_nodes = int(match.group(key))
                    for _ in range(nb_nodes):
                        node_line = file.readline().split()
                        nodeType = NodeType.Leaf if node_line[1] == 'LN' else NodeType.Internal
                        node = Node(
                            node_id=node_line[0],
                            node_type=nodeType,
                            left_child=int(node_line[2]),
                            right_child=int(node_line[3]),
                            split_feature=int(node_line[4]),
                            split_value=float(node_line[5]),
                            depth=int(node_line[6]),
                            classification=int(node_line[7])
                        )
                        trees[tree_id].append(node)
                line = file.readline()

        return nb_features, nb_classes, trees
    
    @staticmethod
    def export(filename, dataset_name, ensemble_type, nb_features, nb_classes, trees, print_content=False):
        with open(filename, 'w') as file:
            file.write("DATASET_NAME: {}\n".format(dataset_name))
            file.write("ENSEMBLE: {}\n".format(ensemble_type))
            file.write("NB_TREES: {}\n".format(len(trees)))
            file.write("NB_FEATURES: {}\n".format(nb_features))
            file.write("NB_CLASSES: {}\n".format(nb_classes))
            file.write("MAX_TREE_DEPTH: {}\n".format(max([node.depth for node in trees[0]])))
            file.write("Format: node / node type(LN - leave node, IN - internal node) left child / right child / feature / threshold / node_depth / majority class (starts with index 0)\n")
            file.write("\n")
            for i, tree in enumerate(trees):
                file.write("[TREE {}]\n".format(i))
                file.write("NB_NODES: {}\n".format(len(tree)))
                for node in tree:
                    file.write("{} ".format(node.node_id))
                    file.write("{} ".format("LN" if node.node_type == NodeType.Leaf else "IN"))
                    file.write("{} ".format(node.left_child))
                    file.write("{} ".format(node.right_child))
                    file.write("{} ".format(node.split_feature))
                    file.write("{} ".format(node.split_value))
                    file.write("{} ".format(node.depth))
                    file.write("{}\n".format(node.classification))
        if print_content:
            with open(filename) as file:
                print(file.read())