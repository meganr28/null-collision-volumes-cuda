#include <iostream>
#include "lbvh.h"
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

Scene::Scene(string filename, GuiParameters& gui_params) {
    std::cout << "Reading scene from " << filename << " ..." << endl;
    std::cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        std::cout << "Error reading from file - aborting!" << endl;
        throw;
    }

    openvdb::initialize();
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1], gui_params);
                std::cout << " " << endl;
            } 
            else if (strcmp(tokens[0].c_str(), "MEDIUM") == 0) {
                loadMedium(tokens[1], gui_params);
                std::cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1], gui_params);
                std::cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                std::cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "ENVMAP") == 0) {
                loadEnvironmentMap(tokens[1]);
                std::cout << " " << endl;
            }
        }
    }

    // Generate LBVH
    if (triangles.size() > 0)
    {
        generateLBVH(this);
    }
}


int Scene::loadEnvironmentMap(string file_name) {
    auto texture = new image(file_name);
    env_map_width = texture->xSize;
    env_map_height = texture->ySize;
    for (int y = 0; y < env_map_height; y++) {
        for (int x = 0; x < env_map_width; x++) {
            int i = y * env_map_width + x;
            //std::cout << texture->pixels[i].x << " " << texture->pixels[i].y << " " << texture->pixels[i].z << std::endl;
        }
    }


    cudaMalloc(&dev_environment_map, texture->xSize * texture->ySize * sizeof(glm::vec3));
    cudaMemcpy(dev_environment_map, texture->pixels, texture->xSize * texture->ySize * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    Light newLight;
    newLight.geom_ID = -1;
    lights.push_back(newLight);

    delete texture;
    return 1;
}

int Scene::loadGeom(string objectid, GuiParameters& gui_params) {
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
        //int starting_tri_size = num_tris;
        int geomTris = 0;
        int starting_tri_size = triangles.size();

        if (newGeom.type == MESH) {

            utilityCore::safeGetline(fp_in, line);
            if (!line.empty() && fp_in.good()) {
                loadOBJ(newGeom, geomTris, line.c_str(), id);
            }
        }

        //int ending_tri_size = mesh_tris.size();
        int ending_tri_size = starting_tri_size + geomTris;

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        if (newGeom.type == MESH) {
            for (int i = starting_tri_size; i < ending_tri_size; ++i) {
                //mesh_tris[i].mat_ID = newGeom.materialid;
                triangles[i].mat_ID = newGeom.materialid;
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
                //mesh_tris[i].mediumInterface = newGeom.mediumInterface;
                triangles[i].mediumInterface = newGeom.mediumInterface;
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

        //if (newGeom.type != MESH) {
            geoms.push_back(newGeom);
            if (materials[newGeom.materialid].emittance > 0.0f) {
                Light newLight;
                newLight.geom_ID = geoms.size() - 1;
                lights.push_back(newLight);
            }
        //}

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

int Scene::loadMaterial(string materialid, GuiParameters& gui_params) {
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

int Scene::loadMedium(string mediumid, GuiParameters& gui_params) {
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

        if (newMedium.type == HOMOGENEOUS) {
            newMedium.aabb_min = glm::vec3(0.0f);
            newMedium.aabb_max = glm::vec3(0.0f);
            newMedium.index_min = glm::vec3(0.0f);
            newMedium.index_max = glm::vec3(0.0f);
            newMedium.gx = 0;
            newMedium.gx = 0;
            newMedium.gz = 0;
            newMedium.maxDensity = 0.1f;
            newMedium.invMaxDensity = 10.0f;
        }

        if (newMedium.type == HETEROGENEOUS) {
            utilityCore::safeGetline(fp_in, line);
            if (!line.empty() && fp_in.good()) {
                // convert from .vdb to .nvdb
                openvdb::io::File file(line.c_str());
                //openvdb::io::File file("../scenes/vdb/cube.vdb");

                file.open();
                openvdb::io::File::NameIterator nameIter = file.beginName();
                std::cout << "Grid name: " << nameIter.gridName() << std::endl;
                
                auto srcGrid = file.readGrid(nameIter.gridName());

                // Convert the OpenVDB grid, srcGrid, into a NanoVDB grid handle.
                gridHandle = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(srcGrid);


                // Define a (raw) pointer to the NanoVDB grid on the host. Note we match the value type of the srcGrid!
                auto* grid = gridHandle.grid<float>();
                if (!grid)
                    throw std::runtime_error("GridHandle does not contain a grid with value type float");

                // Get accessors for the two grids.Note that accessors only accelerate repeated access!
                auto dstAcc = grid->getAccessor();

                std::cout << "Grid count: " << grid->gridCount() << std::endl;

                // Access and print out a cross-section of the narrow-band level set from the two grids
                //for (int i = 0; i < 10; ++i) {
                //    printf("(%3i,0,0) NanoVDB CPU: % 4.2f\n", i, dstAcc.getValue(nanovdb::Coord(i, i, i)));
                //}

                // load attributes into Medium struct and add to media list
                auto boundingBox = grid->worldBBox();
                auto min_index = grid->worldToIndex(boundingBox.min());
                auto max_index = grid->worldToIndex(boundingBox.max());

                auto gridDim = boundingBox.dim();
                nanovdb::Vec3R aabb_min = boundingBox.min();
                nanovdb::Vec3R aabb_max = boundingBox.max();

                newMedium.aabb_min = glm::vec3(aabb_min[0], aabb_min[1], aabb_min[2]);
                newMedium.aabb_max = glm::vec3(aabb_max[0], aabb_max[1], aabb_max[2]);

                newMedium.index_min = glm::vec3(min_index[0], min_index[1], min_index[2]);
                newMedium.index_max = glm::vec3(max_index[0], max_index[1], max_index[2]);
                
                // Cell count in x, y, z
                nanovdb::Vec3f gridExtent = nanovdb::Vec3f(aabb_max - aabb_min) / nanovdb::Vec3f(grid->voxelSize());
                newMedium.gx = gridExtent[0];
                newMedium.gy = gridExtent[1];
                newMedium.gz = gridExtent[2];
                std::cout << "Fog Volume Sphere Min: " << newMedium.aabb_min[0] << " " << newMedium.aabb_min[1] << " " << newMedium.aabb_min[2] << std::endl;
                std::cout << "Fog Volume Sphere Max: " << newMedium.aabb_max[0] << " " << newMedium.aabb_max[1] << " " << newMedium.aabb_max[2] << std::endl;
                std::cout << "Fog Volume Index Min: " << newMedium.index_min[0] << " " << newMedium.index_min[1] << " " << newMedium.index_min[2] << std::endl;
                std::cout << "Fog Volume Index Max: " << newMedium.index_max[0] << " " << newMedium.index_max[1] << " " << newMedium.index_max[2] << std::endl;
                std::cout << "Voxel Size: " << grid->voxelSize()[0] << " " << grid->voxelSize()[1] << " " << grid->voxelSize()[2] << std::endl;
                std::cout << "Fog Volume Sphere Extent: " << gridExtent[0] << " " << gridExtent[1] << " " << gridExtent[2] << std::endl;

                // Set inverse max density
                float maxDensity = 0.0f;
                int numVoxels = newMedium.gx * newMedium.gy * newMedium.gz;
                for (int x = newMedium.index_min.x; x < newMedium.index_max.x; ++x) {
                    for (int y = newMedium.index_min.y; y < newMedium.index_max.y; ++y) {
                        for (int z = newMedium.index_min.z; z < newMedium.index_max.z; ++z) {
                            maxDensity = glm::max(maxDensity, dstAcc.getValue(nanovdb::Coord(x, y, z)));
                        }
                    }
                }
                newMedium.maxDensity = maxDensity;
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
                gui_params.sigma_a = sigma_a;
            }
            else if (strcmp(tokens[0].c_str(), "SCATTERING") == 0) {
                float ss = atof(tokens[1].c_str());
                glm::vec3 sigma_s(ss, ss, ss);
                newMedium.sigma_s = sigma_s;
                gui_params.sigma_s = sigma_s;
            }
            else if (strcmp(tokens[0].c_str(), "ASYM_G") == 0) {
                newMedium.g = atof(tokens[1].c_str());
                gui_params.g = newMedium.g;
            }
        }
#ifdef FULL_VOLUME_INTEGRATOR
        
        newMedium.sigma_t = newMedium.maxDensity * (newMedium.sigma_a + newMedium.sigma_s);
        //newMedium.sigma_n = newMedium.sigma_t - newMedium.sigma_a - newMedium.sigma_s; TODO: find appropriate place to set sigma_n
#else
        newMedium.sigma_t = newMedium.sigma_a + newMedium.sigma_s;
#endif
        media.push_back(newMedium);
        return 1;
    }
}

// Load obj using tinyobjloader (based off of example give by tinyobj library)
int Scene::loadOBJ(Geom &newGeom, int& geomTris, string filename, int objectid)
{
    // Load obj using tinyobjloader
    std::string inputfile = filename;
    tinyobj::ObjReaderConfig reader_config;
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    meshCount = 0;
    for (size_t s = 0; s < shapes.size(); s++) {
        std::vector<Tri> mesh_triangles;

        // Track aabb
        mesh_aabbs.resize(shapes.size());
        glm::vec3 min = glm::vec3(INFINITY, INFINITY, INFINITY);
        glm::vec3 max = glm::vec3(-INFINITY, -INFINITY, -INFINITY);

        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            Tri triangle;

            int i = 0;
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                tinyobj::real_t vx = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                triangle.verts[i] = glm::vec3((float)vx, (float)vy, (float)vz);

                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    triangle.norms[i] = glm::vec3((float)nx, (float)ny, (float)nz);
                }

                // Determine AABB min and max
                min = glm::min(min, triangle.verts[i]);
                max = glm::max(max, triangle.verts[i]);

                i++;
            }
            triangle.computeAABB();
            triangle.computeCentroid();
            triangle.computePlaneNormal();
            triangle.computeArea();
            triangle.objectId = f;
            mesh_triangles.push_back(triangle);

            index_offset += fv;
        }

        // Set AABB
        mesh_aabbs[s].min = min;
        mesh_aabbs[s].max = max;

        // Set mesh attributes
        newGeom.aabb = mesh_aabbs[s];
        newGeom.startIdx = triangles.size();
        newGeom.triangleCount = mesh_triangles.size();
        geomTris = newGeom.triangleCount;
        triangles.insert(triangles.end(), mesh_triangles.begin(), mesh_triangles.end());
        meshCount++;
    }

    return 1;
}