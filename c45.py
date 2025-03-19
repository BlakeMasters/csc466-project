import csv
import json
import pandas as pd
import numpy as np
import os
#Blake Masters
#Lab 2 CSC 466
class c45Node:
    def __init__(self, feature=None, threshold=None, branches=None, *, value=None, probability=None):
        self.feature = feature
        self.threshold = threshold
        self.branches = branches if branches is not None else {}
        self.value = value
        self.probability = probability

    def is_leaf(self):
        return self.value is not None

    def add_branch(self, branch_label, node):
        self.branches[branch_label] = node

    def __repr__(self):
        if self.is_leaf():
            return f"Leaf(class={self.value}, p={self.probability})"
        else:
            if self.threshold is not None:
                return f"Node(feature={self.feature}, threshold={self.threshold}, branches={list(self.branches.keys())})"
            else:
                return f"Node(feature={self.feature}, branches={list(self.branches.keys())})"

    def to_dict(self):
        """(old format). Not used for saving in required format."""
        if self.is_leaf():
            return {"leaf": True, "class": self.value, "p": self.probability}
        else:
            node_dict = {"leaf": False, "feature": self.feature, "branches": {}}
            if self.threshold is not None:
                node_dict["threshold"] = self.threshold
            for branch_label, subtree in self.branches.items():
                node_dict["branches"][str(branch_label)] = subtree.to_dict() if subtree else None
            return node_dict

    def to_json(self, indent=4):
        return json.dumps(self.to_dict(), indent=indent)

class c45:
    def __init__(self, split_metric="Gain", threshold=0.5, attribute_types=None):

        self.threshold = threshold
        self.split_metric = split_metric  #"Gain" or "Ratio"
        self.tree = None
        self.attribute_types = attribute_types if attribute_types is not None else {} #it needs this for numeric vs categorical

    def entropy(self, data, target):
        target_values = data[target]
        unique, counts = np.unique(target_values, return_counts=True)
        probabilities = counts / len(target_values)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def information_gain(self, df, attribute, target, total_entropy):
        values, counts = np.unique(df[attribute], return_counts=True)
        weighted_entropy = 0
        for i, val in enumerate(values):
            subset = df[df[attribute] == val]
            subset_entropy = self.entropy(subset, target)
            weight = counts[i] / len(df)
            weighted_entropy += weight * subset_entropy
        return total_entropy - weighted_entropy

    def information_gain_ratio(self, df, attribute, target, total_entropy):
        gain = self.information_gain(df, attribute, target, total_entropy)
        values, counts = np.unique(df[attribute], return_counts=True)
        probabilities = counts / len(df)
        split_info = -np.sum(probabilities * np.log2(probabilities + 1e-9)) 
        if split_info == 0:
            return 0
        return gain / split_info

    def best_numeric_split(self, df, attribute, target, total_entropy):
        unique_vals = np.sort(df[attribute].astype(float).unique())
        best_gain = -np.inf
        best_threshold = None
        for i in range(len(unique_vals) - 1):
            threshold = (unique_vals[i] + unique_vals[i + 1]) / 2.0
            left = df[df[attribute].astype(float) <= threshold]
            right = df[df[attribute].astype(float) > threshold]
            if left.empty or right.empty:
                continue
            left_entropy = self.entropy(left, target)
            right_entropy = self.entropy(right, target)
            weighted_entropy = (len(left) / len(df)) * left_entropy + (len(right) / len(df)) * right_entropy
            gain = total_entropy - weighted_entropy
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        return best_threshold, best_gain

    def best_numeric_split_ratio(self, df, attribute, target, total_entropy):
        unique_vals = np.sort(df[attribute].astype(float).unique())
        best_ratio = -np.inf
        best_threshold = None
        for i in range(len(unique_vals) - 1):
            threshold = (unique_vals[i] + unique_vals[i + 1]) / 2.0
            left = df[df[attribute].astype(float) <= threshold]
            right = df[df[attribute].astype(float) > threshold]
            if left.empty or right.empty:
                continue
            left_entropy = self.entropy(left, target)
            right_entropy = self.entropy(right, target)
            weighted_entropy = (len(left) / len(df)) * left_entropy + (len(right) / len(df)) * right_entropy
            gain = total_entropy - weighted_entropy
            prob_left = len(left) / len(df)
            prob_right = len(right) / len(df)
            split_info = -(prob_left * np.log2(prob_left + 1e-9) + prob_right * np.log2(prob_right + 1e-9))
            if split_info == 0:
                ratio = 0
            else:
                ratio = gain / split_info
            if ratio > best_ratio:
                best_ratio = ratio
                best_threshold = threshold
        return best_threshold, best_ratio

    def select_splitting_attribute(self, attributes, data, target):
        total_entropy = self.entropy(data, target)
        best_attribute = None
        best_threshold = None
        best_metric = -np.inf
        for attribute in attributes:
            attr_type = self.attribute_types.get(attribute, "categorical")
            if attr_type == "ignore":
                continue
            if self.split_metric == "Gain":
                if attr_type == "numeric":
                    threshold_candidate, gain = self.best_numeric_split(data, attribute, target, total_entropy)
                    if threshold_candidate is not None and gain > best_metric and gain >= self.threshold:
                        best_metric = gain
                        best_attribute = attribute
                        best_threshold = threshold_candidate
                else:
                    gain = self.information_gain(data, attribute, target, total_entropy)
                    if gain > best_metric and gain >= self.threshold:
                        best_metric = gain
                        best_attribute = attribute
                        best_threshold = None
            elif self.split_metric == "Ratio":
                if attr_type == "numeric":
                    threshold_candidate, ratio = self.best_numeric_split_ratio(data, attribute, target, total_entropy)
                    if threshold_candidate is not None and ratio > best_metric and ratio >= self.threshold:
                        best_metric = ratio
                        best_attribute = attribute
                        best_threshold = threshold_candidate
                else:
                    ratio = self.information_gain_ratio(data, attribute, target, total_entropy)
                    if ratio > best_metric and ratio >= self.threshold:
                        best_metric = ratio
                        best_attribute = attribute
                        best_threshold = None
        return best_attribute, best_threshold

    def _build_tree(self, training_set, target):
        outcomes = training_set[target]
        unique_outcomes = np.unique(outcomes)
        attributes = [col for col in training_set.columns if col != target]
        if len(unique_outcomes) == 1:
            count = outcomes.value_counts()[unique_outcomes[0]]
            p = count / len(outcomes)
            return c45Node(value=unique_outcomes[0], probability=p)
        if len(attributes) == 0:
            majority_class = outcomes.mode()[0]
            count = outcomes.value_counts()[majority_class]
            p = count / len(outcomes)
            return c45Node(value=majority_class, probability=p)
        best_attribute, best_threshold = self.select_splitting_attribute(attributes, training_set, target)
        if best_attribute is None:
            majority_class = outcomes.mode()[0]
            count = outcomes.value_counts()[majority_class]
            p = count / len(outcomes)
            return c45Node(value=majority_class, probability=p)
        attr_type = self.attribute_types.get(best_attribute, "categorical")
        if attr_type == "numeric":
            node = c45Node(feature=best_attribute, threshold=best_threshold)
            left_subset = training_set[training_set[best_attribute].astype(float) <= best_threshold]
            right_subset = training_set[training_set[best_attribute].astype(float) > best_threshold]
            if left_subset.empty:
                left_node = c45Node(value=training_set[target].mode()[0],
                                    probability=training_set[target].value_counts()[training_set[target].mode()[0]] / len(training_set[target]))
            else:
                left_node = self._build_tree(left_subset, target)
            if right_subset.empty:
                right_node = c45Node(value=training_set[target].mode()[0],
                                     probability=training_set[target].value_counts()[training_set[target].mode()[0]] / len(training_set[target]))
            else:
                right_node = self._build_tree(right_subset, target)
            node.add_branch("le", left_node)
            node.add_branch("gt", right_node)
        else:
            node = c45Node(feature=best_attribute)
            for attr_value in training_set[best_attribute].unique():
                subset = training_set[training_set[best_attribute] == attr_value].drop(columns=[best_attribute])
                if subset.empty:
                    subtree = c45Node(value=training_set[target].mode()[0],
                                      probability=training_set[target].value_counts()[training_set[target].mode()[0]] / len(training_set[target]))
                else:
                    subtree = self._build_tree(subset, target)
                node.add_branch(attr_value, subtree)
        return node

    def _get_feature_name(self, row, feature):
        if feature in row.index:
            return feature
        lower_to_actual = {col.lower(): col for col in row.index}
        return lower_to_actual.get(feature.lower(), feature)

    def _predict_sample(self, row, node):
        if node.is_leaf():
            return node.value
        if node.threshold is not None:
            feature = self._get_feature_name(row, node.feature)
            try:
                val = float(row[feature])
            except Exception:
                val = row[feature]
            if val <= node.threshold:
                return self._predict_sample(row, node.branches["le"])
            else:
                return self._predict_sample(row, node.branches["gt"])
        else:
            feature = self._get_feature_name(row, node.feature)
            branch = row[feature]
            if branch in node.branches:
                return self._predict_sample(row, node.branches[branch])
            else:
                first_branch = next(iter(node.branches.values()))
                return self._predict_sample(row, first_branch)

    def predict(self, X_test):
        predictions = []
        for idx, row in X_test.iterrows():
            predictions.append(self._predict_sample(row, self.tree))
        return predictions

    """Converts my tree structure to the desired ouput format given by the example"""
    def _node_to_output(self, node):
        if node.is_leaf():
            return {"leaf": {"decision": node.value, "p": node.probability}}
        if node.threshold is not None:
            edges = []
            left_child = node.branches["le"]
            edge_left = {"edge": {"value": node.threshold, "op": "<="}}
            if left_child.is_leaf():
                edge_left["edge"]["leaf"] = {"decision": left_child.value, "p": left_child.probability}
            else:
                edge_left["edge"]["node"] = self._node_to_output(left_child)
            edges.append(edge_left)
            right_child = node.branches["gt"]
            edge_right = {"edge": {"value": node.threshold, "op": ">"}}
            if right_child.is_leaf():
                edge_right["edge"]["leaf"] = {"decision": right_child.value, "p": right_child.probability}
            else:
                edge_right["edge"]["node"] = self._node_to_output(right_child)
            edges.append(edge_right)
            return {"var": node.feature, "edges": edges}
        else:
            edges = []
            for branch_value, child in node.branches.items():
                edge_obj = {"edge": {"value": branch_value}}
                if child.is_leaf():
                    edge_obj["edge"]["leaf"] = {"decision": child.value, "p": child.probability}
                else:
                    edge_obj["edge"]["node"] = self._node_to_output(child)
                edges.append(edge_obj)
            return {"var": node.feature, "edges": edges}

    def get_output_dict(self, dataset_filename):
        return {"dataset": dataset_filename, "node": self._node_to_output(self.tree)}

    def to_output_json(self, dataset_filename, indent=4):
        return json.dumps(self.get_output_dict(dataset_filename), indent=indent)

    def _output_to_tree(self, d):
        if "leaf" in d:
            leaf_info = d["leaf"]
            return c45Node(value=leaf_info["decision"], probability=leaf_info.get("p"))
        node = c45Node(feature=d["var"], threshold=d.get("threshold"))
        for edge_obj in d["edges"]:
            edge = edge_obj["edge"]
            if "op" in edge:
                key = "le" if edge["op"] == "<=" else "gt"
                if "node" in edge:
                    child = self._output_to_tree(edge["node"])
                elif "leaf" in edge:
                    leaf_info = edge["leaf"]
                    child = c45Node(value=leaf_info["decision"], probability=leaf_info.get("p"))
                node.add_branch(key, child)
            else:
                key = edge["value"]
                if "node" in edge:
                    child = self._output_to_tree(edge["node"])
                elif "leaf" in edge:
                    leaf_info = edge["leaf"]
                    child = c45Node(value=leaf_info["decision"], probability=leaf_info.get("p"))
                node.add_branch(key, child)
        return node

    def read_tree(self, filename):
        with open(filename, "r") as f:
            tree_dict = json.load(f)
        if "node" in tree_dict:
            node_dict = tree_dict["node"]
        else:
            node_dict = tree_dict
        self.tree = self._output_to_tree(node_dict)
        print(f"Tree read from {filename}")

    def fit(self, training_set, truth, save=False, output_filename=None, dataset_filename=None):
        self.tree = self._build_tree(training_set, truth)
        if save:
            if output_filename is None or dataset_filename is None:
                print("Error: To save in output format, you must provide output_filename and dataset_filename.")
            else:
                self.save_tree(dataset_filename, output_filename)
        return self.tree

    def save_tree(self, dataset_filename, output_filename):
        tree_dict = self.get_output_dict(dataset_filename)
        with open(output_filename, "w") as f:
            json.dump(tree_dict, f, indent=4)
        #print(f"Tree saved to {output_filename} (output format)")
        return

    def print_tree(self, dataset_filename):
        print(self.to_output_json(dataset_filename, indent=4))

    def get_splits(self):
        """Return a list of all features used to split in this decision tree."""
        
        def dfs_collect_features(node, collected):
            if not node or node.is_leaf():
                return
            collected.append(node.feature)
            for child_node in node.branches.values():
                dfs_collect_features(child_node, collected)
        
        if not self.tree:
            return []  # No tree has been built yet
        splits = []
        dfs_collect_features(self.tree, splits)
        return splits