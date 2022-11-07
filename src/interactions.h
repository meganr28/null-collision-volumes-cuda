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



