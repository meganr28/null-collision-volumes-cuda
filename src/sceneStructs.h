#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include <thrust/random.h>

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define MAX_FLOAT 1000000000.0f

// VOLUME STRUCTS

// Represents a homogeneous medium
struct HomogeneousMedium {
    glm::vec3 sigma_a; // Absorption coefficient
    glm::vec3 sigma_s; // Scattering coefficient
    glm::vec3 sigma_t; // Extinction
    float g;           // Asymmetry factor for Henyey-Greenstein
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
    // plane normal
    glm::vec3 plane_normal;
    float S;
    int mat_ID;
    MediumInterface mediumInterface;
};

struct Geom {
    enum GeomType type;
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
    glm::vec3 accumulatedIrradiance;
    glm::vec3 rayThroughput;
    thrust::default_random_engine rng_engine;
    int remainingBounces;
    int medium;
    bool prev_hit_was_specular;
};

struct MISLightRay {
    Ray ray;
    glm::vec3 f;
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
  glm::vec3 surfaceNormal;
  int materialId;
  MediumInterface mediumInterface;
  MediumInteraction mi;
};


