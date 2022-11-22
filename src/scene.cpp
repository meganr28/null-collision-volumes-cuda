#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_obj_loader.h"
#include <stack>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }

    openvdb::initialize();
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } 
            else if (strcmp(tokens[0].c_str(), "MEDIUM") == 0) {
                loadMedium(tokens[1]);
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

        //link medium interface
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.mediumInterface.inside = atoi(tokens[1].c_str());
            newGeom.mediumInterface.outside = atoi(tokens[2].c_str());
            cout << "Connecting Geom " << objectid << " to Medium Interface " << newGeom.mediumInterface.inside << " " << newGeom.mediumInterface.outside << "..." << endl;
        }

        if (newGeom.type == MESH) {
            for (int i = starting_tri_size; i < ending_tri_size; ++i) {
                mesh_tris[i].mediumInterface = newGeom.mediumInterface;
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

int Scene::loadGLTF(string gltf_file) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltf_file);

    return 0;
}



int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 8; i++) {
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
        else if (strcmp(tokens[0].c_str(), "MEDIUM") == 0) {
            camera.medium = atoi(tokens[1].c_str());
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

int Scene::loadMedium(string mediumid) {
    int id = atoi(mediumid.c_str());
    if (id != media.size()) {
        cout << "ERROR: MEDIUM ID does not match expected number of media" << endl;
        return -1;
    }
    else {
        cout << "Loading Medium " << id << "..." << endl;
        Medium newMedium;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "homogeneous") == 0) {
                cout << "Creating new homogeneous volume..." << endl;
                newMedium.type = HOMOGENEOUS;
            }
            else if (strcmp(line.c_str(), "heterogeneous") == 0) {
                cout << "Creating new heterogeneous volume..." << endl;
                newMedium.type = HETEROGENEOUS;
            }
        }

        if (newMedium.type == HETEROGENEOUS) {
            utilityCore::safeGetline(fp_in, line);
            if (!line.empty() && fp_in.good()) {
                // convert from .vdb to .nvdb
                openvdb::io::File file(line.c_str());
                //openvdb::io::File file("../scenes/vdb/cube.vdb");

                file.open();
                openvdb::io::File::NameIterator nameIter = file.beginName();
                auto srcGrid = file.readGrid(nameIter.gridName());

                // Convert the OpenVDB grid, srcGrid, into a NanoVDB grid handle.
                gridHandle = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(srcGrid);

                // Define a (raw) pointer to the NanoVDB grid on the host. Note we match the value type of the srcGrid!
                auto* grid = gridHandle.grid<float>();
                if (!grid)
                    throw std::runtime_error("GridHandle does not contain a grid with value type float");

                // Get accessors for the two grids.Note that accessors only accelerate repeated access!
                auto dstAcc = grid->getAccessor();
                // Access and print out a cross-section of the narrow-band level set from the two grids
                //for (int i = 0; i < 10; ++i) {
                //    printf("(%3i,0,0) NanoVDB CPU: % 4.2f\n", i, dstAcc.getValue(nanovdb::Coord(i, i, i)));
                //}

                // load attributes into Medium struct and add to media list
                auto boundingBox = grid->worldBBox();
                auto gridDim = boundingBox.dim();
                nanovdb::Vec3R aabb_min = boundingBox.min();
                nanovdb::Vec3R aabb_max = boundingBox.max();

                newMedium.aabb_min = glm::vec3(aabb_min[0], aabb_min[1], aabb_min[2]);
                newMedium.aabb_max = glm::vec3(aabb_max[0], aabb_max[1], aabb_max[2]);
                
                // Cell count in x, y, z
                nanovdb::Vec3f gridExtent = nanovdb::Vec3f(aabb_max - aabb_min) / nanovdb::Vec3f(grid->voxelSize());
                newMedium.gx = gridExtent[0];
                newMedium.gy = gridExtent[1];
                newMedium.gz = gridExtent[2];
                std::cout << "Fog Volume Sphere Min: " << newMedium.aabb_min[0] << " " << newMedium.aabb_min[1] << " " << newMedium.aabb_min[2] << std::endl;
                std::cout << "Fog Volume Sphere Max: " << newMedium.aabb_max[0] << " " << newMedium.aabb_max[1] << " " << newMedium.aabb_max[2] << std::endl;
                //std::cout << "Voxel Size: " << grid->voxelSize()[0] << " " << grid->voxelSize()[1] << " " << grid->voxelSize()[2] << std::endl;
                //std::cout << "Fog Volume Sphere Extent: " << gridExtent[0] << " " << gridExtent[1] << " " << gridExtent[2] << std::endl;

                // Set inverse max density
                float maxDensity = 0.0f;
                int numVoxels = newMedium.gx * newMedium.gy * newMedium.gz;
                for (int x = 0; x < newMedium.gx; ++x) {
                    for (int y = 0; y < newMedium.gy; ++y) {
                        for (int z = 0; z < newMedium.gz; ++z) {
                            maxDensity = glm::max(maxDensity, dstAcc.getValue(nanovdb::Coord(x, y, z)));
                        }
                    }
                }   
                newMedium.invMaxDensity = 1.0f / maxDensity;
                std::cout << "Max Density: " << maxDensity << std::endl;
                std::cout << "Inv Max Density: " << newMedium.invMaxDensity << std::endl;

                // TODO: add translate, rotate, scale to Medium specification in scene file (mediumToWorld, inverse() will get worldTomedium)
                nanovdb::Vec3f diff = nanovdb::Vec3f(aabb_max - aabb_min);
                glm::vec3 scl = glm::vec3(diff[0], diff[1], diff[2]);
                glm::mat4 scale_matrix = glm::scale(glm::mat4(), scl);

                glm::vec3 trans = glm::vec3(aabb_min[0], aabb_min[1], aabb_min[2]);
                glm::mat4 translate_matrix = glm::translate(glm::mat4(), trans);

                newMedium.worldToMedium = glm::inverse(translate_matrix * scale_matrix);
                std::cout << "World To Medium: " << glm::to_string(newMedium.worldToMedium) << std::endl;

                glm::vec4 aabb_min_v4 = glm::vec4(aabb_min[0], aabb_min[1], aabb_min[2], 1.0);
                glm::vec4 aabb_max_v4 = glm::vec4(aabb_max[0], aabb_max[1], aabb_max[2], 1.0);
                glm::vec3 transformed_aabb_min = glm::vec3(newMedium.worldToMedium * aabb_min_v4);
                glm::vec3 transformed_aabb_max = glm::vec3(newMedium.worldToMedium * aabb_max_v4);

                /*std::cout << "Transformed Min Before: " << glm::to_string(aabb_min_v4) << std::endl;
                std::cout << "Transformed Min After: " << glm::to_string(transformed_aabb_min) << std::endl;

                std::cout << "Transformed Max Before: " << glm::to_string(aabb_max_v4) << std::endl;
                std::cout << "Transformed Max After: " << glm::to_string(transformed_aabb_max) << std::endl;*/

                file.close();
            }
        }

        //load static properties
        for (int i = 0; i < 3; i++) {
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "ABSORPTION") == 0) {
                float sa = atof(tokens[1].c_str());
                glm::vec3 sigma_a(sa, sa, sa);
                newMedium.sigma_a = sigma_a;
            }
            else if (strcmp(tokens[0].c_str(), "SCATTERING") == 0) {
                float ss = atof(tokens[1].c_str());
                glm::vec3 sigma_s(ss, ss, ss);
                newMedium.sigma_s = sigma_s;
            }
            else if (strcmp(tokens[0].c_str(), "ASYM_G") == 0) {
                newMedium.g = atof(tokens[1].c_str());
            }
        }
        newMedium.sigma_t = newMedium.sigma_a + newMedium.sigma_s;
        media.push_back(newMedium);
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