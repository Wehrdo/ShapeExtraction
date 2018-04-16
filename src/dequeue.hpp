#pragma once
#include "cuda_runtime.h"


template <typename T>
class Q_Node {
public:
    const T* node;
    float priority;
};

template <typename T, int MAX_Q>
class PriorityQueue {
private:
    Q_Node<T> data[MAX_Q]; // Array to hold heap values
    __host__ __device__ void percolateDown(Q_Node<T> item, int idx);
    __host__ __device__ void percolateDownMin(Q_Node<T> item, int idx);
    __host__ __device__ void percolateDownMax(Q_Node<T> item, int idx);
    __host__ __device__ void PriorityQueue::bubbleUp(int idx);
    __host__ __device__ void PriorityQueue::bubbleUpMin(int idx);
    __host__ __device__ void PriorityQueue::bubbleUpMax(int idx);
    __host__ __device__ int parent(const int i) {
        return (i - 1) / 2;
    }
    __host__ __device__ int childLeft(const int i) {
        return 2 * i + 1;
    }
    __host__ __device__ int childRight(const int i) {
        return 2 * i + 2;
    }
    __host__ __device__ void PriorityQueue::swap(int idx1, int idx2);
    __host__ __device__ int largest_idx();
public:
    // Number of items contained
    int size;
    __host__ __device__ PriorityQueue(): size(0) {}
    // Inserts a new item into the queue
    __host__ __device__ void insert(const T* node, float weight);
    // Returns and removes the item with lowest value
    __host__ __device__ Q_Node<T> removeMin();
    // Returns and removes the item with highest value
    __host__ __device__ Q_Node<T> removeMax();
    __host__ __device__ Q_Node<T> peekMax();

};