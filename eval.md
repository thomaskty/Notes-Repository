| **Property**                | **Silhouette Score**                                               | **DB Score**                                                   | **CH Score**                                                  |

|-----------------------------|-------------------------------------------------------------------|---------------------------------------------------------------|--------------------------------------------------------------|

|**Main Goal**| Measures how well each point lies within its cluster and how distinct clusters are. | Evaluates compactness within clusters and separation between clusters. | Measures the ratio of between-cluster dispersion to within-cluster dispersion. |

|**Range of Values**|[-1, 1] (Higher is better, with 1 indicating perfect clustering) | [0, ∞) (Lower is better, with 0 indicating perfect clustering) | [0, ∞) (Higher is better, with larger values indicating better-defined clusters) |

|**Compactness**| Assesses how close points are to their cluster centroid.         | Directly penalizes clusters with high intra-cluster variance. | Indirectly considers compactness via within-cluster variance. |

|**Separation**| Measures how distinct clusters are from each other.              | Considers the average distance between cluster centroids.     | Focuses on the dispersion between cluster centroids.         |

|**Cluster Shape Assumptions**| Works well for convex clusters but can handle some irregular shapes. | Assumes convex and isotropic clusters (not suitable for irregular shapes). | Prefers convex, spherical clusters (sensitive to shape assumptions). |

|**Scaling Sensitivity**| Sensitive to data scaling; normalization is required.            | Sensitive to scaling; normalization is essential.             | Sensitive to scaling; data should be standardized.           |

|**Number of Clusters**| Penalizes having too few or too many clusters.                   | May favor larger numbers of clusters, leading to overestimation. | Favors solutions with well-separated clusters, regardless of the number. |

|**Impact of Noise/Outliers**| Outliers reduce the score significantly (handles them moderately well). | Highly sensitive to outliers, as they affect centroids and distances. | Moderate sensitivity; outliers inflate within-cluster variance. |

|**Applicability to Non-Euclidean Metrics**| Works with non-Euclidean metrics (e.g., cosine distance).      | Generally requires a Euclidean distance metric.               | Requires a Euclidean metric for meaningful results.          |

|**Interpretation Simplicity**| Intuitive and interpretable.                                      | Less intuitive, relies on averages of compactness/separation. | More abstract, based on dispersion ratios.                   |

|**Use in Dimensionality Reduction**| Suitable for understanding clustering quality after PCA.          | May be less effective in reduced dimensions.                  | Effective if clusters remain well-separated post-reduction.  |

|**Algorithm Independence**| Algorithm-agnostic, applicable to all clustering methods.        | Algorithm-agnostic, but assumes centroids represent clusters. | Algorithm-agnostic, works well with centroid-based methods.  |

|**Preferred Scenarios**| Best for well-separated, moderately compact clusters.            | Works for spherical, evenly spread clusters.                  | Best for clusters with clear dispersion differences.          |

|**Code Implementation**|`sklearn.metrics.silhouette_score`| Custom computation or `sklearn.metrics.davies_bouldin_score`|`sklearn.metrics.calinski_harabasz_score`|
