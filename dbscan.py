
import pandas as pd
import numpy as np
import sys
import csv
import random
import math
from collections import defaultdict
"""
Lab 4 csc 466
Blake Masters
bemaster@calpoly.edu
David Cho
dcho08@calpoly.edu
"""
def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        print("Error: Empty file.")
        sys.exit(1)

    restrictions = [int(x.strip()) for x in rows[0]]
    num_cols = len(restrictions)
    used_indices = [i for i, flag in enumerate(restrictions) if flag == 1]
    ground_truth_index = None   
    if restrictions[-1] == 0:
        ground_truth_index = num_cols - 1

    data = []
    ground_truth = [] if ground_truth_index is not None else None
    for row in rows[1:]:
        row = [field for field in row if field.strip() != '']
        if len(row) != num_cols:
            continue
        try:
            point = [float(row[i].strip()) for i in used_indices]
        except ValueError:
            print("Error: Non-numeric value encountered in feature columns.")
            sys.exit(1)
        data.append(point)
        if ground_truth_index is not None:
            ground_truth.append(row[ground_truth_index].strip())

    return np.array(data), ground_truth

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def build_distance_matrix(data):
    n = len(data)
    distances = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dist = euclidean_distance(data[i], data[j])
            distances[i][j] = dist
            distances[j][i] = dist
    return distances

def build_neighbor_lists(data, distances, epsilon):
    n = len(data)
    neighbors = []
    for i in range(n):
        neighbor_indices = []
        for j in range(n):
            if distances[i][j] <= epsilon:
                neighbor_indices.append(j)
        neighbors.append(neighbor_indices)
    return neighbors

def dbscan(data, epsilon, min_points):
    """
    returns
    cluster_labels: a list of cluster indices for each point (or -1 if outlier)
    num_clusters: how many clusters found
    """
    n = len(data)
    #-1 for outliers, 0 means unvisited, >0 means cluster label
    cluster_labels = [0]*n  
    visited = [False]*n  
    current_cluster_label = 0
    
    #Precompute distances and neighbor lists
    distances = build_distance_matrix(data)
    neighbors = build_neighbor_lists(data, distances, epsilon)

    for i in range(n):
        if not visited[i]:
            visited[i] = True
            neighbor_i = neighbors[i]
            if len(neighbor_i) < min_points:
                #noise
                cluster_labels[i] = -1
            else:
                #new cluster
                current_cluster_label += 1
                cluster_labels[i] = current_cluster_label
                #expand cluster
                expand_cluster(i, neighbor_i, neighbors, cluster_labels, visited, current_cluster_label, min_points)
    
    return cluster_labels, current_cluster_label

def expand_cluster(point_index, neighbor_indices, neighbors, cluster_labels, visited, current_cluster_label, min_points):
    #manage the points to process
    queue = list(neighbor_indices)
    
    while queue:
        neighbor = queue.pop()
        
        if not visited[neighbor]:
            visited[neighbor] = True
            neighbor_list = neighbors[neighbor]
            if len(neighbor_list) >= min_points:
                # Merge this neighbor's neighbors with the queue
                for x in neighbor_list:
                    if cluster_labels[x] == 0:  
                        queue.append(x)
                        
        #neighbor unassigned (cluster_labels[neighbor] == 0) or is noise (-1),
        #assign it to the current cluster
        if cluster_labels[neighbor] == 0 or cluster_labels[neighbor] == -1:
            cluster_labels[neighbor] = current_cluster_label

def compute_cluster_stats(data, cluster_labels):
    

    clusters = defaultdict(list)
    for i, c in enumerate(cluster_labels):
        if c != -1:
            clusters[c].append(i)
    
    outliers = [i for i, c in enumerate(cluster_labels) if c == -1]
    cluster_stats = {}
    for c_label, indices in clusters.items():
        #centroid
        coords = list(zip(*(data[i] for i in indices)))  # transpose
        centroid = [sum(dim)/len(dim) for dim in coords]  
        #dist to centroid
        distances = [euclidean_distance(data[i], centroid) for i in indices]
        min_d = min(distances)
        max_d = max(distances)
        avg_d = sum(distances)/len(distances)
        
        sse = sum(d**2 for d in distances)
        
        cluster_stats[c_label] = {
            "points": indices,
            "centroid": centroid,
            "min_dist": min_d,
            "max_dist": max_d,
            "avg_dist": avg_d,
            "sse": sse,
            "num_points": len(indices)
        }
    
    return cluster_stats, outliers

def main():
    if len(sys.argv) < 4:
        print("Usage: python dbscan.py <Filename> <epsilon> <NumPoints>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        epsilon = float(sys.argv[2])
        min_points = int(sys.argv[3])
    except ValueError:
        print("Error: <epsilon> must be a float and <NumPoints> must be an integer.")
        sys.exit(1)

    data, ground_truth = load_data(filename)
    print(f"Loaded {len(data)} data points with {data.shape[1]} feature(s).")

    cluster_labels, num_clusters = dbscan(data, epsilon, min_points)
    cluster_stats, outliers = compute_cluster_stats(data, cluster_labels)

    for c_label, stats in sorted(cluster_stats.items()):
        print(f"\nCluster {c_label}:")
        print(f"  Centroid: {stats['centroid']}")
        print(f"  Number of points: {stats['num_points']}")
        print(f"  Max distance to centroid: {stats['max_dist']}")
        print(f"  Min distance to centroid: {stats['min_dist']}")
        print(f"  Avg distance to centroid: {stats['avg_dist']}")
        print(f"  Sum of Squared Errors (SSE): {stats['sse']}")
        print(f"  Points: {', '.join(map(str, stats['points']))}")
    print("\nOutliers:")
    total_outliers = len(outliers)
    percentage = (total_outliers / len(data)) * 100 if data.size > 0 else 0
    print(f"  Total number of outliers: {total_outliers}")
    print(f"  Percentage of dataset: {percentage:.2f}%")
    if len(data) < 50:
        print("  Outlier indices: " + ", ".join(map(str, outliers)))

if __name__ == '__main__':
    main()