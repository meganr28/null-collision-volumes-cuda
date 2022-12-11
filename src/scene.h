#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "image.h"

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include "../external/include/openvdb/nanovdb/nanovdb/util//CudaDeviceBuffer.h"
#include "../external/include/openvdb/nanovdb/nanovdb/util/IO.h"
#include "../external/include/openvdb/nanovdb/nanovdb/util/OpenToNanoVDB.h"

//#include "../external/include/tiny_gltf.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid, GuiParameters& gui_params);
    int loadMedium(string mediumid, GuiParameters& gui_params);
    int loadGeom(string objectid, GuiParameters& gui_params);
    int loadGLTF(string objectid);
    int loadEnvironmentMap(string file_name);
    int loadCamera();


public:

    Scene(string filename, GuiParameters& gui_params);
    ~Scene();

    int loadOBJ(Geom& newGeom, int& geomTris, string filename, int objectid);

    int num_tris = 0;

    std::vector<Geom> geoms;
    int num_geoms = 0;
    //std::vector<Mesh> meshes;
    std::vector<Light> lights;
    std::vector<Material> materials;
    std::vector<Medium> media;
    
    // Handle to heterogeneous volume grid data
    nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> gridHandle;

    std::vector<Tri> mesh_tris;
    std::vector<Tri> mesh_tris_sorted;

    // LBVH arrays
    std::vector<Tri> triangles;
    std::vector<Tri> sorted_triangles;
    std::vector<MortonCode> mcodes;
    std::vector<LBVHNode> lbvh;
    std::vector<AABB> mesh_aabbs;
    int meshCount;

    RenderState state;

    int environment_map_ID;
    glm::vec3* dev_environment_map = NULL;
    float* dev_env_map_distribution = NULL;
    int env_map_width = 0;
    int env_map_height = 0;
    float env_map_dist_sum = 0.0f;
};
