#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadMedium(string mediumid);
    int loadGeom(string objectid);
    int loadCamera();


public:

    Scene(string filename);
    ~Scene();

    BVHNode* buildBVH(int start_index, int end_index);
    void reformatBVHToGPU();

    int num_tris = 0;

    std::vector<Geom> geoms;
    int num_geoms = 0;
    //std::vector<Mesh> meshes;
    std::vector<Light> lights;
    std::vector<Material> materials;
    std::vector<HomogeneousMedium> media;

    std::vector<Tri> mesh_tris;
    std::vector<Tri> mesh_tris_sorted;

    BVHNode* root_node;
    int num_nodes = 0;

    std::vector<BVHNode_GPU> bvh_nodes_gpu;
    std::vector<TriBounds> tri_bounds;
    RenderState state;
};
