#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define MIN_INTERSECT_DIST 0.0001f
#define MAX_INTERSECT_DIST 10000.0f

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
inline __host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

inline __device__ float squareplaneIntersectionTest(Geom& squareplane, Ray& r, glm::vec3& normal) {
    Ray q;
    q.origin = glm::vec3(squareplane.inverseTransform * glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(glm::vec3(squareplane.inverseTransform * glm::vec4(r.direction, 0.0f)));

    float t = glm::dot(glm::vec3(0.0f, 0.0f, 1.0f), (glm::vec3(0.5f, 0.5f, 0.0f) - q.origin)) / glm::dot(glm::vec3(0.0f, 0.0f, 1.0f), q.direction);
    glm::vec3 objspaceIntersection = q.origin + t * q.direction;

    if (t > 0.0001f && objspaceIntersection.x >= -0.5001f && objspaceIntersection.x <= 0.5001f && objspaceIntersection.y >= -0.5001f && objspaceIntersection.y <= 0.5001f) {
        normal = glm::normalize(glm::vec3(squareplane.invTranspose * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));
        return glm::length(r.origin - glm::vec3(squareplane.transform * glm::vec4(objspaceIntersection, 1.0f)));
    }

    return MAX_INTERSECT_DIST;
}


// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
inline __device__ float boxIntersectionTest(Geom &box, Ray &r, glm::vec3 &normal) {
    Ray q;
    q.origin    = glm::vec3(box.inverseTransform *glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(glm::vec3(box.inverseTransform * glm::vec4(r.direction, 0.0f)));

    float tmin = -MAX_INTERSECT_DIST;
    float tmax = MAX_INTERSECT_DIST;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    
    float t1;
    float t2;
    float ta;
    float tb;
#pragma unroll
    for (int xyz = 0; xyz < 3; ++xyz) {
        t1 = (-0.5f - q.origin[xyz]) / q.direction[xyz];
        t2 = (+0.5f - q.origin[xyz]) / q.direction[xyz];
        if (t1 < t2) {
            ta = t1;
            tb = t2;
        }
        else {
            tb = t1;
            ta = t2;
        }
        glm::vec3 n;
        n[xyz] = t2 < t1 ? +1.0f : -1.0f;
        if (ta > 0.0f && ta > tmin) {
            tmin = ta;
            tmin_n = n;
        }
        if (tb < tmax) {
            tmax = tb;
            tmax_n = n;
        }
    }

    if (tmax >= tmin && tmax > 0.0f) {
        if (tmin <= 0.0f) {
            tmin = tmax;
            tmin_n = tmax_n;
        }
        normal = glm::normalize(glm::vec3(box.invTranspose * glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - glm::vec3(box.transform * glm::vec4(q.origin + tmin * q.direction, 1.0f)));
    }
    return MAX_INTERSECT_DIST;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
inline __device__ float sphereIntersectionTest(Geom &sphere, Ray &r, glm::vec3 &normal) {

    glm::vec3 ro = glm::vec3(sphere.inverseTransform * glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(glm::vec3(sphere.inverseTransform * glm::vec4(r.direction, 0.0f)));

    float vDotDirection = glm::dot(ro, rd);
    float radicand_and_t = vDotDirection * vDotDirection - (glm::dot(ro, ro) - 0.25f);
    if (radicand_and_t < 0.0f) {
        return MAX_INTERSECT_DIST;
    }

    float squareRoot = glm::sqrt(radicand_and_t);
    float t1 = -vDotDirection + squareRoot;
    float t2 = -vDotDirection - squareRoot;

    if (t1 < 0.0f && t2 < 0.0f) {
        return MAX_INTERSECT_DIST;
    } 
    else if (t1 > 0.0f && t2 > 0.0f) {
        radicand_and_t = glm::min(t1, t2);
    } 
    else {
        radicand_and_t = glm::max(t1, t2);
    }

    glm::vec3 objspaceIntersection = ro + radicand_and_t * rd;

    normal = glm::normalize(glm::vec3(sphere.invTranspose * glm::vec4(objspaceIntersection, 0.0f)));

    return glm::length(r.origin - glm::vec3(sphere.transform * glm::vec4(objspaceIntersection, 1.0f)));
}

inline __host__ __device__ bool aabbIntersectionTest(const glm::vec3& aabbMin, const glm::vec3& aabbMax, Ray& r, float& tMin, float& tMax, float& t) {
    float x1 = (aabbMin.x - r.origin.x) * r.direction_inv.x;
    float x2 = (aabbMax.x - r.origin.x) * r.direction_inv.x;

    tMin = glm::min(x1, x2);
    tMax = glm::max(x1, x2);

    float y1 = (aabbMin.y - r.origin.y) * r.direction_inv.y;
    float y2 = (aabbMax.y - r.origin.y) * r.direction_inv.y;

    tMin = glm::max(tMin, glm::min(y1, y2));
    tMax = glm::min(tMax, glm::max(y1, y2));

    float z1 = (aabbMin.z - r.origin.z) * r.direction_inv.z;
    float z2 = (aabbMax.z - r.origin.z) * r.direction_inv.z;

    tMin = glm::max(tMin, glm::min(z1, z2));
    tMax = glm::min(tMax, glm::max(z1, z2));

    bool intersect = tMin <= tMax && tMax >= 0;
    t = (intersect) ? tMin : -1.0;
    if (t < 0.0f) t = tMax;

    return intersect;
}
