import numpy as np
import matplotlib.pyplot as plt

def plot_clusters_with_rewards(rewards, clusters, padding=0.5):
    unique_clusters = np.unique(clusters)
    colors = plt.cm.get_cmap("tab20", len(unique_clusters))

    fig, ax = plt.subplots(figsize=(7, 7))

    for i, cluster in enumerate(unique_clusters):
        cluster_points = rewards[clusters == cluster, :2]
        color = colors(i)

        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color)

        center = cluster_points.mean(axis=0)

        max_distance = np.max(np.linalg.norm(cluster_points - center, axis=1)) + padding

        circle = plt.Circle(center, max_distance, color=color, alpha=0.2)
        ax.add_patch(circle)

    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.set_title("Clusters com Recompensas")
    plt.grid(True)
    plt.axis("equal")

    return ax

    
def plot_agents_paths(ax, rewards, paths, distance_mx, budget):
    colors = ['blue', 'orange']
    for agent_idx, path in enumerate(paths):
        total_distance = 0
        agent_color = colors[agent_idx % len(colors)]
        
        first_point = rewards[path[0], :2]
        total_distance += np.linalg.norm(first_point)
        
        ax.plot([0, first_point[0]], [0, first_point[1]], color=agent_color, marker='o', markersize=4)
        
        for i in range(1, len(path)):
            start_idx = path[i - 1]
            end_idx = path[i]
            segment_distance = distance_mx[start_idx, end_idx]
            
            if total_distance + segment_distance > budget:
                break
            
            start_point = rewards[start_idx, :2]
            end_point = rewards[end_idx, :2]
            
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                    color=agent_color, linewidth=2, marker='o', markersize=4)
            
            total_distance += segment_distance

    plt.show()