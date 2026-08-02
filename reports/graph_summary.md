# Spatial Graph Summary

## Overview

This report summarizes the K-Nearest Neighbor (KNN) spatial graph constructed from the engineered King County Housing dataset.

Each property is represented as a graph node, while edges connect geographically nearby properties.

## Graph Statistics

- Nodes: **21,613**
- Edges: **216,130**
- K Nearest Neighbours: **10**

## Edge Distance Statistics

- Minimum Distance: **0.0000**
- Average Distance: **0.0031**
- Maximum Distance: **0.2994**

## Validation

- Graph successfully generated.
- Node mapping exported.
- Edge index exported.
- Edge distances exported.
- Degree validation passed.
- Ready for Graph Neural Network training.

## Next Stage

The generated graph will be transformed into a PyTorch Geometric Data object. Node features, edge indices and regression targets will then be used to train the Graph Neural Network during Week 4.
