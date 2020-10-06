// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "octree.h"
#include "utils.h"
#include <utility> 
#include <chrono>
using namespace std::chrono; 


typedef struct OcTree
{
    int depth;
    int index;
    at::Tensor center;
    struct OcTree *children[8];
    void init(at::Tensor center, int d, int i) {
        this->center = center;
        this->depth = d;
        this->index = i;
        for (int i=0; i<8; i++) this->children[i] = nullptr;
    }
}OcTree;

class EasyOctree {
    public:
        OcTree *root;
        int total;
        int terminal;

        at::Tensor all_centers;
        at::Tensor all_children;

        EasyOctree(at::Tensor center, int depth) {
            root = new OcTree;
            root->init(center, depth, -1);
            total = -1;
            terminal = -1;
        }
        ~EasyOctree() {
            OcTree *p = root;
            destory(p);
        }
        void destory(OcTree * &p);
        void insert(OcTree * &p, at::Tensor point, int index);
        void finalize();
        std::pair<int, int> count(OcTree * &p);
};

void EasyOctree::destory(OcTree * &p){
    if (p != nullptr) {
        for (int i=0; i<8; i++) {
            if (p->children[i] != nullptr) destory(p->children[i]);
        }
        delete p;
        p = nullptr;
    }
}

void EasyOctree::insert(OcTree * &p, at::Tensor point, int index) {
    at::Tensor diff = (point > p->center).to(at::kInt);
    int idx = diff[0].item<int>() + 2 * diff[1].item<int>() + 4 * diff[2].item<int>();
    if (p->depth == 0) {
        p->children[idx] = new OcTree;
        p->children[idx]->init(point, -1, index);
    } else {
        if (p->children[idx] == nullptr) {
            int length = 1 << (p->depth - 1);
            at::Tensor new_center = p->center + (2 * diff - 1) * length;
            p->children[idx] = new OcTree;
            p->children[idx]->init(new_center, p->depth-1, -1);
        }
        insert(p->children[idx], point, index);
    }
}

std::pair<int, int> EasyOctree::count(OcTree * &p) {
    int total = 0, terminal = 0;
    for (int i=0; i<8; i++) {
        if (p->children[i] != nullptr) {
            std::pair<int, int> sub = count(p->children[i]);
            total += sub.first;
            terminal += sub.second;
        }
    }
    total += 1;
    if (p->depth == -1) terminal += 1;
    return std::make_pair(total, terminal);
}

void EasyOctree::finalize() {
    std::pair<int, int> outs = count(root);
    total = outs.first; terminal = outs.second;
    
    all_centers =
      torch::zeros({outs.first, 3}, at::device(root->center.device()).dtype(at::ScalarType::Int));
    all_children =
      -torch::ones({outs.first, 9}, at::device(root->center.device()).dtype(at::ScalarType::Int));

    int node_idx = outs.first - 1;
    root->index = node_idx;

    std::queue<OcTree*> all_leaves; all_leaves.push(root);    
    while (!all_leaves.empty()) {
        OcTree* node_ptr = all_leaves.front();
        all_leaves.pop();
        for (int i=0; i<8; i++) {
            if (node_ptr->children[i] != nullptr) {
                if (node_ptr->children[i]->depth > -1) {
                    node_idx--; 
                    node_ptr->children[i]->index = node_idx;
                }
                all_leaves.push(node_ptr->children[i]);
                all_children[node_ptr->index][i] = node_ptr->children[i]->index;
            }
        }
        all_children[node_ptr->index][8] = 1 << (node_ptr->depth + 1);
        all_centers[node_ptr->index] = node_ptr->center;
    }
    assert (node_idx == outs.second);
};

std::tuple<at::Tensor, at::Tensor> build_octree(at::Tensor center, at::Tensor points, int depth) {
    auto start = high_resolution_clock::now();
    EasyOctree tree(center, depth);
    for (int k=0; k<points.size(0); k++) 
        tree.insert(tree.root, points[k], k);
    tree.finalize();
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start);
    printf("Building EasyOctree done. total #nodes = %d, terminal #nodes = %d (time taken %f s)\n", 
        tree.total, tree.terminal, float(duration.count()) / 1000000.);
    return std::make_tuple(tree.all_centers, tree.all_children);
}