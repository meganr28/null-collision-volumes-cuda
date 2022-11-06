#pragma once

#include "intersections.h"
#include <thrust/random.h>


//#define USE_SCHLICK_APPROX

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
inline __host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng, thrust::uniform_real_distribution<float>& u01) {

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

// Using information from PBRT
// https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission
// and https://en.wikipedia.org/wiki/Schlick%27s_approximation
// and https://en.wikipedia.org/wiki/Fresnel_equations

inline __device__ glm::vec3 fresnelDielectric(float cos_theta_i, float etaT) {
    
    // assume scene medium is air
    float etaI = 1.0f;

    cos_theta_i = glm::clamp(cos_theta_i, -1.0f, 1.0f);

    // flip the indices of refraction if exiting the medium
    if (cos_theta_i > 0.0f) {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        
    }
    cos_theta_i = glm::abs(cos_theta_i);

#ifdef USE_SCHLICK_APPROX
    // schlicks approximation

    float r = (etaI - etaT) / (etaI + etaT);
    float R_0 = r * r;
    float Fr = R_0 + (1.0f - R_0) * glm::pow((1.0f - cos_theta_i), 5.0f);
    return glm::vec3(Fr);
#else
    // physically based fresnel term

    // need to solve for cos(thetaT)
    // use sin^2 = 1 - cos^2
    float sinThetaI = glm::sqrt(1.0f - cos_theta_i * cos_theta_i);

    // snells law sin(thetaT) = n1/n2 * sin(thetaI)
    float sinThetaT = etaI / etaT * sinThetaI;

    // total internal reflection
    if (sinThetaT >= 0.999f) {
        return glm::vec3(1.0f);
    }

    float cos_theta_t = glm::sqrt(1.0f - sinThetaT * sinThetaT);

    float eta_t_cos_theta_i = etaT * cos_theta_i;
    float eta_i_cos_theta_i = etaI * cos_theta_i;
    float eta_t_cos_theta_t = etaT * cos_theta_t;
    float eta_i_cos_theta_t = etaI * cos_theta_t;

    // light polarization equations
    float Rparl = ((eta_t_cos_theta_i) - (eta_i_cos_theta_t)) / ((eta_t_cos_theta_i) + (eta_i_cos_theta_t));
    float Rperp = ((eta_i_cos_theta_i) - (eta_t_cos_theta_t)) / ((eta_i_cos_theta_i) + (eta_t_cos_theta_t));
    return glm::vec3((Rparl * Rparl + Rperp * Rperp) / 2.0f);
#endif
}

inline __device__
void scatterRay(
        PathSegment& r,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng,
    thrust::uniform_real_distribution<float>& u01) {

    glm::vec3 wi = glm::vec3(0.0f);
    glm::vec3 f = glm::vec3(0.0f);
    float pdf = 0.0f;
    float absDot = 0.0f;

    //thrust::uniform_real_distribution<float> u01(0, 1);

    // Physically based BSDF sampling influenced by PBRT
    // https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission
    // https://www.pbr-book.org/3ed-2018/Reflection_Models/Lambertian_Reflection

    if (m.type == SPEC_BRDF) {
        wi = glm::reflect(r.ray.direction, normal);
        absDot = glm::abs(glm::dot(normal, wi));
        pdf = 1.0f;
        if (absDot >= -0.0001f && absDot <= -0.0001f) {
            f = m.R;
        }
        else {
            f = m.R / absDot;
        }
    }
    else if (m.type == SPEC_BTDF) {
        // spec refl
        float eta = m.ior;
        if (glm::dot(normal, r.ray.direction) < 0.0001f) {
            // outside
            eta = 1.0f / eta;
            wi = glm::refract(r.ray.direction, normal, eta);
        }
        else {
            // inside
            wi = glm::refract(r.ray.direction, -normal, eta);
        }
        absDot = glm::abs(glm::dot(normal, wi));
        pdf = 1.0f;
        if (glm::length(wi) <= 0.0001f) {
            // total internal reflection
            f = glm::vec3(0.0f);
        }
        else if (absDot >= -0.0001f && absDot <= -0.0001f) {
            f = m.T;
        }
        else {
            f = m.T / absDot;
        }
    }
    else if (m.type == SPEC_GLASS) {
        // spec glass
        float eta = m.ior;
        if (u01(rng) < 0.5f) {
            // spec refl
            wi = glm::reflect(r.ray.direction, normal);
            absDot = glm::abs(glm::dot(normal, wi));
            pdf = 1.0f;
            if (absDot == 0.0f) {
                f = m.R;
            }
            else {
                f = m.R / absDot;
            }
            f *= fresnelDielectric(glm::dot(normal, r.ray.direction), m.ior);
        }
        else {
            // spec refr
            if (glm::dot(normal, r.ray.direction) < 0.0f) {
                // outside
                eta = 1.0f / eta;
                wi = glm::refract(r.ray.direction, normal, eta);
            }
            else {
                // inside
                wi = glm::refract(r.ray.direction, -normal, eta);
            }
            absDot = glm::abs(glm::dot(normal, wi));
            pdf = 1.0f;
            if (glm::length(wi) <= 0.0001f) {
                // total internal reflection
                f = glm::vec3(0.0f);
            }
            if (absDot == 0.0f) {
                f = m.T;
            }
            else {
                f = m.T / absDot;
            }
            f *= glm::vec3(1.0f) - fresnelDielectric(glm::dot(normal, r.ray.direction), m.ior);
        }
        f *= 2.0f;
    }
    else {
        // diffuse
        wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng, u01));
        absDot = glm::abs(glm::dot(normal, wi));
        pdf = absDot * 0.31831f;
        f = m.R * 0.31831f;
    }

    r.rayThroughput *=  f * absDot / pdf;

    // Change ray direction
    r.ray.direction = wi;
    r.ray.direction_inv = glm::vec3(1.0f / wi.x, 1.0f / wi.y, 1.0f / wi.z);
    r.ray.origin = intersect + (wi * 0.001f);
}

