#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_obj_loader.h"
#include <stack>

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } 
            else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }

    if (mesh_tris.size() > 0) {
        root_node = buildBVH(0, mesh_tris.size());

        reformatBVHToGPU();

        std::cout << "num nodes: " << num_nodes << std::endl;
    }

    /*for (int i = 0; i < num_nodes; ++i) {
        std::cout << "NODE " << i << std::endl;
        std::cout << "   AABB_min: " << bvh_nodes_gpu[i].AABB_min.x << " " << bvh_nodes_gpu[i].AABB_min.y << " " << bvh_nodes_gpu[i].AABB_min.z << " " << std::endl;
        std::cout << "   AABB_max: " << bvh_nodes_gpu[i].AABB_max.x << " " << bvh_nodes_gpu[i].AABB_max.y << " " << bvh_nodes_gpu[i].AABB_max.z << " " << std::endl;
        std::cout << "   num_tris: " << bvh_nodes_gpu[i].num_tris << std::endl;
        std::cout << "   tri_index: " << bvh_nodes_gpu[i].tri_index << std::endl;
        std::cout << "   offset_to_second_child: " << bvh_nodes_gpu[i].offset_to_second_child << std::endl;
        std::cout << "   split_axis: " << bvh_nodes_gpu[i].axis << std::endl;
    }*/


}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != num_geoms) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        num_geoms++;
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            else if (strcmp(line.c_str(), "squareplane") == 0) {
                cout << "Creating new squareplane..." << endl;
                newGeom.type = SQUAREPLANE;
            }
            else if (strcmp(line.c_str(), "mesh") == 0) {
                std::cout << "Creating new mesh..." << std::endl;
                newGeom.type = MESH;
            }
        }

        std::cout << num_tris << std::endl;
        int starting_tri_size = num_tris;

        if (newGeom.type == MESH) {

            utilityCore::safeGetline(fp_in, line);
            if (!line.empty() && fp_in.good()) {
                tinyobj::attrib_t attrib;
                std::vector<tinyobj::shape_t> shapes;
                std::vector<tinyobj::material_t> materials;
                std::string warn, err;

                if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, line.c_str())) {
                    throw std::runtime_error(warn + err);
                }

                // every mesh in the obj
                for (const tinyobj::shape_t& shape : shapes) {



                    // every tri in the mesh
                    for (int i = 0; i < shape.mesh.indices.size(); i += 3) {
                        Tri newTri;

                        

                        glm::vec3 newP = glm::vec3(0.0f);
                        glm::vec3 newN = glm::vec3(0.0f);
                        glm::vec2 newT = glm::vec2(0.0f);

                        for (int k = 0; k < 3; ++k) {

                            if (shape.mesh.indices[i + k].vertex_index != -1) {
                                newP = glm::vec3(attrib.vertices[3 * shape.mesh.indices[i + k].vertex_index + 0],
                                    attrib.vertices[3 * shape.mesh.indices[i + k].vertex_index + 1],
                                    attrib.vertices[3 * shape.mesh.indices[i + k].vertex_index + 2]);
                            }

                            if (shape.mesh.indices[i + k].texcoord_index != -1) {
                                newT = glm::vec2(
                                    attrib.texcoords[2 * shape.mesh.indices[i + k].texcoord_index + 0],
                                    1.0f - attrib.texcoords[2 * shape.mesh.indices[i + k].texcoord_index + 1]
                                );
                            }

                            if (shape.mesh.indices[i + k].normal_index != -1) {
                                newN = glm::vec3(
                                    attrib.normals[3 * shape.mesh.indices[i + k].normal_index + 0],
                                    attrib.normals[3 * shape.mesh.indices[i + k].normal_index + 1],
                                    attrib.normals[3 * shape.mesh.indices[i + k].normal_index + 2]
                                );
                            }

                            if (k == 0) {
                                newTri.p0 = newP;
                                newTri.n0 = newN;
                                newTri.t0 = newT;
                            }
                            else if (k == 1) {
                                newTri.p1 = newP;
                                newTri.n1 = newN;
                                newTri.t1 = newT;
                            }
                            else {
                                newTri.p2 = newP;
                                newTri.n2 = newN;
                                newTri.t2 = newT;
                            }
                        }


                        newTri.plane_normal = glm::normalize(glm::cross(newTri.p1 - newTri.p0, newTri.p2 - newTri.p1));
                        newTri.S = glm::length(glm::cross(newTri.p1 - newTri.p0, newTri.p2 - newTri.p1));

                        TriBounds newTriBounds;

                        newTriBounds.tri_ID = num_tris;
                        

                        float max_x = glm::max(glm::max(newTri.p0.x, newTri.p1.x), newTri.p2.x);
                        float max_y = glm::max(glm::max(newTri.p0.y, newTri.p1.y), newTri.p2.y);
                        float max_z = glm::max(glm::max(newTri.p0.z, newTri.p1.z), newTri.p2.z);
                        newTriBounds.AABB_max = glm::vec3(max_x, max_y, max_z);

                        float min_x = glm::min(glm::min(newTri.p0.x, newTri.p1.x), newTri.p2.x);
                        float min_y = glm::min(glm::min(newTri.p0.y, newTri.p1.y), newTri.p2.y);
                        float min_z = glm::min(glm::min(newTri.p0.z, newTri.p1.z), newTri.p2.z);
                        newTriBounds.AABB_min = glm::vec3(min_x, min_y, min_z);

                        float mid_x = (newTri.p0.x + newTri.p1.x + newTri.p2.x) / 3.0;
                        float mid_y = (newTri.p0.y + newTri.p1.y + newTri.p2.y) / 3.0;
                        float mid_z = (newTri.p0.z + newTri.p1.z + newTri.p2.z) / 3.0;
                        newTriBounds.AABB_centroid = glm::vec3(mid_x, mid_y, mid_z);

                        tri_bounds.push_back(newTriBounds);

                        mesh_tris.push_back(newTri);
                        num_tris++;
                    }
                    //std::cout << num_tris << std::endl;
                }
            }
        }

        int ending_tri_size = mesh_tris.size();

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        if (newGeom.type == MESH) {
            for (int i = starting_tri_size; i < ending_tri_size; ++i) {
                mesh_tris[i].mat_ID = newGeom.materialid;
            }
        }
        

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        if (newGeom.type != MESH) {
            geoms.push_back(newGeom);
            if (materials[newGeom.materialid].emittance > 0.0f) {
                Light newLight;
                newLight.geom_ID = geoms.size() - 1;
                lights.push_back(newLight);
            }
        }

        

        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 7; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
        else if (strcmp(tokens[0].c_str(), "FOCAL_DISTANCE") == 0) {
            camera.focal_distance = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "LENS_RADIUS") == 0) {
            camera.lens_radius = atof(tokens[1].c_str());
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 5; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "R_COLOR") == 0) {
                glm::vec3 rColor( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.R = rColor;
            } 
            else if (strcmp(tokens[0].c_str(), "T_COLOR") == 0) {
                glm::vec3 tColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.T = tColor;
            } 
            else if (strcmp(tokens[0].c_str(), "DIFFUSE_BRDF") == 0) {
                newMaterial.type = DIFFUSE_BRDF;
            }
            else if (strcmp(tokens[0].c_str(), "DIFFUSE_BTDF") == 0) {
                newMaterial.type = DIFFUSE_BTDF;
            }
            else if (strcmp(tokens[0].c_str(), "SPEC_BRDF") == 0) {
                newMaterial.type = SPEC_BRDF;
            }
            else if (strcmp(tokens[0].c_str(), "SPEC_BTDF") == 0) {
                newMaterial.type = SPEC_BTDF;
            }
            else if (strcmp(tokens[0].c_str(), "SPEC_GLASS") == 0) {
                newMaterial.type = SPEC_GLASS;
            }
            else if (strcmp(tokens[0].c_str(), "SPEC_PLASTIC") == 0) {
                newMaterial.type = SPEC_PLASTIC;
            }
            else if (strcmp(tokens[0].c_str(), "IOR") == 0) {
                newMaterial.ior = atof(tokens[1].c_str());
            } 
            else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

// PBRT BVH as reference
// https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies

BVHNode* Scene::buildBVH(int start_index, int end_index) {
    BVHNode* new_node = new BVHNode();
    num_nodes++;
    int num_tris_in_node = end_index - start_index;

    // get the AABB bounds for this node (getting min and max of all triangles within)
    glm::vec3 max_bounds = glm::vec3(-100000.0);
    glm::vec3 min_bounds = glm::vec3(100000.0);
    for (int i = start_index; i < end_index; ++i) {
        if (max_bounds.x < tri_bounds[i].AABB_max.x) {
            max_bounds.x = tri_bounds[i].AABB_max.x;
        }
        if (max_bounds.y < tri_bounds[i].AABB_max.y) {
            max_bounds.y = tri_bounds[i].AABB_max.y;
        }
        if (max_bounds.z < tri_bounds[i].AABB_max.z) {
            max_bounds.z = tri_bounds[i].AABB_max.z;
        }

        if (min_bounds.x > tri_bounds[i].AABB_min.x) {
            min_bounds.x = tri_bounds[i].AABB_min.x;
        }
        if (min_bounds.y > tri_bounds[i].AABB_min.y) {
            min_bounds.y = tri_bounds[i].AABB_min.y;
        }
        if (min_bounds.z > tri_bounds[i].AABB_min.z) {
            min_bounds.z = tri_bounds[i].AABB_min.z;
        }
    }

    // leaf node (with 1 tri in it)
    if (num_tris_in_node <= 1) {
        mesh_tris_sorted.push_back(mesh_tris[tri_bounds[start_index].tri_ID]);
        new_node->tri_index = mesh_tris_sorted.size() - 1;
        new_node->AABB_max = max_bounds;
        new_node->AABB_min = min_bounds;
        return new_node;
    }
    // intermediate node (covering tris start_index through end_index
    else {
        // get the greatest length between tri centroids in each direction x, y, and z
        glm::vec3 centroid_max = glm::vec3(-100000.0);
        glm::vec3 centroid_min = glm::vec3(100000.0);
        for (int i = start_index; i < end_index; ++i) {
            if (centroid_max.x < tri_bounds[i].AABB_centroid.x) {
                centroid_max.x = tri_bounds[i].AABB_centroid.x;
            }
            if (centroid_max.y < tri_bounds[i].AABB_centroid.y) {
                centroid_max.y = tri_bounds[i].AABB_centroid.y;
            }
            if (centroid_max.z < tri_bounds[i].AABB_centroid.z) {
                centroid_max.z = tri_bounds[i].AABB_centroid.z;
            }

            if (centroid_min.x > tri_bounds[i].AABB_centroid.x) {
                centroid_min.x = tri_bounds[i].AABB_centroid.x;
            }
            if (centroid_min.y > tri_bounds[i].AABB_centroid.y) {
                centroid_min.y = tri_bounds[i].AABB_centroid.y;
            }
            if (centroid_min.z > tri_bounds[i].AABB_centroid.z) {
                centroid_min.z = tri_bounds[i].AABB_centroid.z;
            }
        }
        glm::vec3 centroid_extent = centroid_max - centroid_min;

        // choose dimension to split along (dimension with largest extent)
        int dimension_to_split = 0;
        if (centroid_extent.x >= centroid_extent.y && centroid_extent.x >= centroid_extent.z) {
            dimension_to_split = 0;
        }
        else if (centroid_extent.y >= centroid_extent.x && centroid_extent.y >= centroid_extent.z) {
            dimension_to_split = 1;
        }
        else {
            dimension_to_split = 2;
        }


        int mid_point = (start_index + end_index) / 2;
        float centroid_midpoint = (centroid_min[dimension_to_split] + centroid_max[dimension_to_split]) / 2;

        if (centroid_min[dimension_to_split] == centroid_max[dimension_to_split]) {
            mesh_tris_sorted.push_back(mesh_tris[tri_bounds[start_index].tri_ID]);
            new_node->tri_index = mesh_tris_sorted.size() - 1;
            new_node->AABB_max = max_bounds;
            new_node->AABB_min = min_bounds;
            return new_node;
        }

        // partition triangles in bounding box, ones with centroids less than the midpoint go before ones with greater than
        // using std::partition for partition algorithm
        // https://en.cppreference.com/w/cpp/algorithm/partition
        TriBounds* pointer_to_partition_point = std::partition(&tri_bounds[start_index], &tri_bounds[end_index - 1] + 1,
                [dimension_to_split, centroid_midpoint](const TriBounds& triangle_AABB) {
                return triangle_AABB.AABB_centroid[dimension_to_split] < centroid_midpoint;
        });

        // get the pointer relative to the start of the array
        mid_point = pointer_to_partition_point - &tri_bounds[0];

        // create two children nodes each for one side of the partitioned node
        new_node->child_nodes[0] = buildBVH(start_index, mid_point);
        new_node->child_nodes[1] = buildBVH(mid_point, end_index);

        new_node->split_axis = dimension_to_split;
        new_node->tri_index = -1;
            
        new_node->AABB_max.x = glm::max(new_node->child_nodes[0]->AABB_max.x, new_node->child_nodes[1]->AABB_max.x);
        new_node->AABB_max.y = glm::max(new_node->child_nodes[0]->AABB_max.y, new_node->child_nodes[1]->AABB_max.y);
        new_node->AABB_max.z = glm::max(new_node->child_nodes[0]->AABB_max.z, new_node->child_nodes[1]->AABB_max.z);

        new_node->AABB_min.x = glm::min(new_node->child_nodes[0]->AABB_min.x, new_node->child_nodes[1]->AABB_min.x);
        new_node->AABB_min.y = glm::min(new_node->child_nodes[0]->AABB_min.y, new_node->child_nodes[1]->AABB_min.y);
        new_node->AABB_min.z = glm::min(new_node->child_nodes[0]->AABB_min.z, new_node->child_nodes[1]->AABB_min.z);
        return new_node;
    }
}

void Scene::reformatBVHToGPU() {
    BVHNode *cur_node;
    std::stack<BVHNode*> nodes_to_process;
    std::stack<int> index_to_parent;
    std::stack<bool> second_child_query;
    int cur_node_index = 0;
    int parent_index = 0;
    bool is_second_child = false;
    nodes_to_process.push(root_node);
    index_to_parent.push(-1);
    second_child_query.push(false);
    while (!nodes_to_process.empty()) {
        BVHNode_GPU new_gpu_node;

        cur_node = nodes_to_process.top();
        nodes_to_process.pop();
        parent_index = index_to_parent.top();
        index_to_parent.pop();
        is_second_child = second_child_query.top();
        second_child_query.pop();

        if (is_second_child && parent_index != -1) {
            bvh_nodes_gpu[parent_index].offset_to_second_child = bvh_nodes_gpu.size();
        }
        new_gpu_node.AABB_min = cur_node->AABB_min;
        new_gpu_node.AABB_max = cur_node->AABB_max;
        if (cur_node->tri_index != -1) {
            // leaf node
            new_gpu_node.tri_index = cur_node->tri_index;
        }
        else {
            // intermediate node
            new_gpu_node.tri_index = -1;
            nodes_to_process.push(cur_node->child_nodes[1]);
            index_to_parent.push(bvh_nodes_gpu.size());
            second_child_query.push(true);
            nodes_to_process.push(cur_node->child_nodes[0]);
            index_to_parent.push(-1);
            second_child_query.push(false);
        }
        bvh_nodes_gpu.push_back(new_gpu_node);
    }
}