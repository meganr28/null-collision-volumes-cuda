#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include <thrust/random.h>

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define MAX_FLOAT 1000000000.0f

// VOLUME STRUCTS

enum IntegratorType {
    NULL_SCATTERING_MIS,
    DELTA_TRACKING_NEE,
    SURFACE_ONLY_MIS
};

enum ImportanceSampling {
    UNI_NEE_MIS,
    NEE,
    UNI
};

enum MediumType {
    HOMOGENEOUS,
    HETEROGENEOUS,
};

// General representation of a medium
struct Medium {
    MediumType type;         // Homogeneous or heterogeneous
    glm::mat4 worldToMedium; // Transform to local medium space
    glm::vec3 aabb_min;      // Minimum of aabb
    glm::vec3 aabb_max;      // Maximum of aabb
    glm::vec3 index_min;  // transformation from world to index space
    glm::vec3 index_max;  // transformation from world to index space
    glm::vec3 sigma_a;       // Absorption coefficient
    glm::vec3 sigma_s;       // Scattering coefficient
    glm::vec3 sigma_t;       // Extinction
    int gx, gy, gz;          // Grid dimensions
    float g;                 // Asymmetry factor for Henyey-Greenstein
    float maxDensity;
    float invMaxDensity; 
};

struct MediumInteraction {
    glm::vec3 samplePoint;  // Point between ray origin and surface interaction
    glm::vec3 wo;           // Outgoing ray direction
    int medium;             // Pointer to medium
};

// Represents possible transition between two mediums
struct MediumInterface {  
    int inside;        // Index to medium on inside
    int outside;       // Index to medium on outside
};

// SCENE STRUCTS

enum GeomType {
    SPHERE,
    CUBE,
    SQUAREPLANE,
    MESH,
    TRI,
};

enum BSDF {
    DIFFUSE_BRDF,
    DIFFUSE_BTDF,
    SPEC_BRDF,
    SPEC_BTDF,
    SPEC_GLASS,
    SPEC_PLASTIC,
    MIRCROFACET_BRDF,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    glm::vec3 direction_inv;
};

struct TriBounds {
    glm::vec3 AABB_min;
    glm::vec3 AABB_max;
    glm::vec3 AABB_centroid;
    int tri_ID;
};

struct MortonCode {
    int objectId;
    unsigned int code;
};

struct NodeRange {
    int i;
    int j;
    int l;
    int d;
};

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

struct LBVHNode {
    AABB aabb;
    int objectId;
    unsigned int left;
    unsigned int right;
};

struct BVHNode {
    glm::vec3 AABB_min;
    glm::vec3 AABB_max;
    BVHNode* child_nodes[2];
    int split_axis;
    int tri_index;
};

struct BVHNode_GPU {
    glm::vec3 AABB_min;
    glm::vec3 AABB_max;
    int tri_index;
    int offset_to_second_child;
};

struct Tri {
    // positions
    glm::vec3 p0;
    glm::vec3 p1;
    glm::vec3 p2;
    // normals
    glm::vec3 n0;
    glm::vec3 n1;
    glm::vec3 n2;
    // uvs
    glm::vec2 t0;
    glm::vec2 t1;
    glm::vec2 t2;
    // array versions
    glm::vec3 verts[3];
    glm::vec3 norms[3];
    // plane normal
    glm::vec3 plane_normal;
    // centroid
    glm::vec3 centroid;
    float S;
    int objectId;
    int mat_ID;
    MediumInterface mediumInterface;
    AABB aabb;

    void computeAABB() {
        aabb.min = glm::min(verts[0], glm::min(verts[1], verts[2]));
        aabb.max = glm::max(verts[0], glm::max(verts[1], verts[2]));
    }

    void computeCentroid() {
        centroid = (verts[0] + verts[1] + verts[2]) / glm::vec3(3.f, 3.f, 3.f);
    }

    void computePlaneNormal() {
        plane_normal = glm::normalize(glm::cross(verts[1] - verts[0], verts[2] - verts[1]));
    }

    void computeArea() {
        S = glm::length(glm::cross(verts[1] - verts[0], verts[2] - verts[1]));
    }
};

struct Geom {
    enum GeomType type;
    AABB aabb;
    int startIdx;
    int triangleCount;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    MediumInterface mediumInterface;
};

struct Light {
    int geom_ID;
};

struct Material {
    glm::vec3 R;
    glm::vec3 T;
    BSDF type;
    float ior;
    float emittance;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float focal_distance;
    float lens_radius;
    int medium;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    Ray lastRealRay;
    glm::vec3 accumulatedIrradiance;
    glm::vec3 rayThroughput;
    glm::vec3 r_u;
    glm::vec3 r_l;
    thrust::default_random_engine rng_engine;
    int pixelIndex;
    int remainingBounces;
    int realPathLength;
    int medium;
    int rgbWavelength;
    bool prev_hit_was_specular;
    bool prev_hit_null_material;
    bool prev_event_was_real;
};

struct MISLightRay {
    Ray ray;
    glm::vec3 f;
    glm::vec3 r_l;
    glm::vec3 r_u;
    float pdf;
    int light_ID;
    int medium;
};

struct MISLightIntersection {
    glm::vec3 LTE;
    float w;
    MediumInterface mediumInterface;
    MediumInteraction mi;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  float tMin; // for bounding box intersections for heterogeneous volumes: TODO REMOVE IF BETTER SOLUTION FOUND!!!
  float tMax;
  glm::vec3 surfaceNormal;
  int objID;
  int materialId;
  MediumInterface mediumInterface;
  MediumInteraction mi;

};

struct SceneInfo {
    int geoms_size;
    int media_size;
    int lights_size;
};

struct GuiParameters {
    glm::vec3 sigma_a;
    glm::vec3 sigma_s;
    float g;
    int max_depth;
    int depth_padding;
    int refresh_rate;
    int refresh_bit;
    float density_offset;
    float density_scale;
    ImportanceSampling importance_sampling;
};

