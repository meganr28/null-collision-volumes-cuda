#pragma once

#include "intersections.h"
#include <thrust/random.h>

#include "../external/include/openvdb/nanovdb/nanovdb/util//CudaDeviceBuffer.h"


//#define USE_SCHLICK_APPROX
#define EPSILON  0.00000001f

enum MediumEvent {
    ABSORB,
    REAL_SCATTER,
    NULL_SCATTER
};

inline __host__ __device__ 
void buildOrthonormalBasis(const glm::vec3& normal, glm::vec3* tangent, glm::vec3* bitangent) {
    if (glm::abs(normal.x) > glm::abs(normal.y))
        *tangent = glm::vec3(-normal.z, 0, normal.x) /
        glm::sqrt(normal.x * normal.x + normal.z * normal.z);
    else
        *tangent = glm::vec3(0, normal.z, -normal.y) /
        glm::sqrt(normal.y * normal.y + normal.z * normal.z);
    *bitangent = glm::cross(normal, *tangent);
}

inline __host__ __device__
glm::vec3 convertToSphericalDirection(float sinTheta, float cosTheta, float phi, const glm::vec3 &x, const glm::vec3& y, const glm::vec3& z) {
    return sinTheta * glm::cos(phi) * x + sinTheta * glm::sin(phi) * y + cosTheta * z;
}

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

inline __host__ __device__ glm::vec3 fresnelDielectric(float cos_theta_i, float etaT) {
    
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

inline __host__ __device__
float evaluatePhaseHG(const glm::vec3& wo, const glm::vec3& wi, float g, float gui_g)
{
    g = gui_g;
    float cosTheta = glm::dot(wo, wi);
    float denom = 1.0f + g * g + 2.0f * g * cosTheta;
    return INV_FOUR_PI * (1.0f - g * g) / (denom * glm::sqrt(denom));
}

inline __host__ __device__
float Sample_p(const glm::vec3& wo, glm::vec3* wi, float* pdf, const glm::vec2& u, float g, float gui_g)
{
    g = gui_g;
    // cosTheta for Henyey-Greenstein
    float cosTheta;
    if (glm::abs(g) < 0.0001) {
        cosTheta = 1 * 2 * u[0];
    }
    else {
        float sqrTerm = (1 - g * g) / (1 - g + 2 * g * u[0]);
        cosTheta = (1 + g * g - sqrTerm * sqrTerm) / (2 * g);
    }

    // Compute wi
    float sinTheta = glm::sqrt(glm::max(0.f, 1 - cosTheta * cosTheta));
    float phi = 2 * PI * u[1];

    // Create orthonormal basis
    glm::vec3 v1, v2;
    buildOrthonormalBasis(wo, &v1, &v2);
    *wi = convertToSphericalDirection(sinTheta, cosTheta, phi, v1, v2, wo);

    float p = evaluatePhaseHG(wo, *wi, g, gui_g);

    if (pdf) {
        *pdf = p;
    }

    return p;
}

inline __host__ __device__
glm::vec3 Sample_f(
    const Material& mat, 
    const PathSegment& segment, 
    const ShadeableIntersection& intersection, 
    glm::vec3* wi, 
    float* pdf,
    float& absDot,
    thrust::default_random_engine& rng,
    thrust::uniform_real_distribution<float>& u01)
{
    // Physically based BSDF sampling influenced by PBRT
    // https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission
    // https://www.pbr-book.org/3ed-2018/Reflection_Models/Lambertian_Reflection

    glm::vec3 f = glm::vec3(0.0f);

    if (mat.type == SPEC_BRDF) {
        *wi = glm::reflect(segment.ray.direction, intersection.surfaceNormal);
        absDot = glm::abs(glm::dot(intersection.surfaceNormal, *wi));
        *pdf = 1.0f;
        if (absDot >= -0.0001f && absDot <= -0.0001f) {
            f = mat.R;
        }
        else {
            f = mat.R / absDot;
        }
    }
    else if (mat.type == SPEC_BTDF) {
        // spec refl
        float eta = mat.ior;
        if (glm::dot(intersection.surfaceNormal, segment.ray.direction) < 0.0001f) {
            // outside
            eta = 1.0f / eta;
            *wi = glm::refract(segment.ray.direction, intersection.surfaceNormal, eta);
        }
        else {
            // inside
            *wi = glm::refract(segment.ray.direction, -intersection.surfaceNormal, eta);
        }
        absDot = glm::abs(glm::dot(intersection.surfaceNormal, *wi));
        *pdf = 1.0f;
        if (glm::length(*wi) <= 0.0001f) {
            // total internal reflection
            f = glm::vec3(0.0f);
        }
        else if (absDot >= -0.0001f && absDot <= -0.0001f) {
            f = mat.T;
        }
        else {
            f = mat.T / absDot;
        }
    }
    else if (mat.type == SPEC_GLASS) {
        // spec glass
        float eta = mat.ior;
        if (u01(rng) < 0.5f) {
            // spec refl
            *wi = glm::reflect(segment.ray.direction, intersection.surfaceNormal);
            absDot = glm::abs(glm::dot(intersection.surfaceNormal, *wi));
            *pdf = 1.0f;
            if (absDot == 0.0f) {
                f = mat.R;
            }
            else {
                f = mat.R / absDot;
            }
            f *= fresnelDielectric(glm::dot(intersection.surfaceNormal, segment.ray.direction), mat.ior);
        }
        else {
            // spec refr
            if (glm::dot(intersection.surfaceNormal, segment.ray.direction) < 0.0f) {
                // outside
                eta = 1.0f / eta;
                *wi = glm::refract(segment.ray.direction, intersection.surfaceNormal, eta);
            }
            else {
                // inside
                *wi = glm::refract(segment.ray.direction, -intersection.surfaceNormal, eta);
            }
            absDot = glm::abs(glm::dot(intersection.surfaceNormal, *wi));
            *pdf = 1.0f;
            if (glm::length(*wi) <= 0.0001f) {
                // total internal reflection
                f = glm::vec3(0.0f);
            }
            if (absDot == 0.0f) {
                f = mat.T;
            }
            else {
                f = mat.T / absDot;
            }
            f *= glm::vec3(1.0f) - fresnelDielectric(glm::dot(intersection.surfaceNormal, segment.ray.direction), mat.ior);
        }
        f *= 2.0f;
    }
    else {
        // diffuse
        *wi = glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng, u01));
        if (mat.type == DIFFUSE_BTDF) {
            *wi = -*wi;
        }
        absDot = glm::abs(glm::dot(intersection.surfaceNormal, *wi));
        *pdf = absDot * 0.31831f;
        f = mat.R * 0.31831f;
    }

    return f;
}

inline __host__ __device__ glm::vec3 getSigmaT(const Medium& medium, GuiParameters& gui_params) {
    return gui_params.sigma_a + gui_params.sigma_s;
}

inline __host__ __device__ float D_heterogeneous(const Medium& medium, const nanovdb::NanoGrid<float>* media_density, glm::vec3 sample_index, PathSegment& segment)
{
    // read density value from the grid
    if (sample_index.x < medium.index_min.x || sample_index.x >= medium.index_max.x ||
        sample_index.y < medium.index_min.y || sample_index.y >= medium.index_max.y ||
        sample_index.z < medium.index_min.z || sample_index.z >= medium.index_max.z) {
        return 0.0f;
    }
        
    auto gpuAcc = media_density->getAccessor();
    auto density = gpuAcc.getValue(nanovdb::Coord(sample_index.x, sample_index.y, sample_index.z));
    return (density >= 0.0f) ? density : -density;
}

inline __host__ __device__ float Density_heterogeneous(const Medium& medium, const nanovdb::NanoGrid<float>* media_density, glm::vec3 sample_point, PathSegment& segment)
{
    // find the sample point's voxel
    glm::vec3 pSamples(sample_point.x * medium.gx - 0.5f, sample_point.y * medium.gy - 0.5f, sample_point.z * medium.gz - 0.5f);
    pSamples += medium.index_min;
    glm::vec3 pi = glm::floor(pSamples);
    glm::vec3 d = pSamples - pi;
    
    //return D_heterogeneous(medium, media_density, pi, segment);

    // trilinear sampling of density values nearest to sampling point
    float d00 = glm::mix(D_heterogeneous(medium, media_density, pi, segment), D_heterogeneous(medium, media_density, pi + glm::vec3(1, 0, 0), segment), d.x);
    float d10 = glm::mix(D_heterogeneous(medium, media_density, pi + glm::vec3(0, 1, 0), segment), D_heterogeneous(medium, media_density, pi + glm::vec3(1, 1, 0), segment), d.x);
    float d01 = glm::mix(D_heterogeneous(medium, media_density, pi + glm::vec3(0, 0, 1), segment), D_heterogeneous(medium, media_density, pi + glm::vec3(1, 0, 1), segment), d.x);
    float d11 = glm::mix(D_heterogeneous(medium, media_density, pi + glm::vec3(0, 1, 1), segment), D_heterogeneous(medium, media_density, pi + glm::vec3(1, 1, 1), segment), d.x);
    float d0 = glm::mix(d00, d10, d.y);
    float d1 = glm::mix(d01, d11, d.y);
    return glm::mix(d0, d1, d.z);
}

inline __host__ __device__ float Density_homogeneous(const Medium& medium)
{
    return medium.maxDensity;
}

inline __host__ __device__ glm::vec3 Tr_homogeneous(const Medium& medium, const Ray& ray, float t)
{
    return glm::exp(-medium.sigma_t * glm::min(t * glm::length(ray.direction), MAX_FLOAT));
}

inline __host__ __device__ glm::vec3 Tr_heterogeneous(
    const Medium& medium, 
    PathSegment& segment, // TODO: Remove this from all functions after debugging
    const MISLightRay& mis_ray, 
    const nanovdb::NanoGrid<float>* media_density,
    float t,
    GuiParameters& gui_params,
    thrust::default_random_engine& rng,
    thrust::uniform_real_distribution<float>& u01)
{
    // Transform ray to local medium space
    Ray worldRay = mis_ray.ray;

    Ray localRay;
    localRay.origin = glm::vec3(medium.worldToMedium * glm::vec4(worldRay.origin, 1.0f));
    localRay.direction = glm::vec3(medium.worldToMedium * glm::vec4(worldRay.direction, 0.0f));
    localRay.direction_inv = 1.0f / localRay.direction;
    float rayTMax = t * glm::length(worldRay.direction); // TODO: use rayTMax in computation?

    // Compute tmin and tmax of ray overlap with medium bounds
    glm::vec3 localBBMin = glm::vec3(0.0f);
    glm::vec3 localBBMax = glm::vec3(1.0f);
    float tMin, tMax, t_intersect;
    if (!aabbIntersectionTest(segment, localBBMin, localBBMax, localRay, tMin, tMax, t_intersect, false)) {
        return glm::vec3(1.0f);
    }
    
    int num_iters = 0;
    float Tr = 1.0f;
    t = glm::max(tMin, 0.0f);
    glm::vec3 samplePoint = localRay.origin + t * localRay.direction;
    while (true) {
        t -= glm::log(1.0f - u01(rng)) * medium.invMaxDensity / getSigmaT(medium, gui_params)[0];  // TODO: sigma_t is a float for heterogeneous medium
        if (t >= tMax) {
            break;
        }

        samplePoint = localRay.origin + t * localRay.direction;
        float density = Density_heterogeneous(medium, media_density, samplePoint, segment);
        Tr *= 1.0 - glm::max(0.0f, density * medium.invMaxDensity);
        num_iters++;
    }
    return glm::vec3(Tr);
}

inline __host__ __device__
glm::vec3 Sample_homogeneous(
    const Medium& medium, 
    const PathSegment& segment, 
    const ShadeableIntersection& isect, 
    MediumInteraction* mi, 
    int mediumIndex, 
    float rand)
{
    const Ray& ray = segment.ray;

    // TODO: change this to randomly select channel with spectral rendering
    int channel = 0;
    float dist = -glm::log(1.0f - rand) / medium.sigma_t[channel];
    float t = glm::min(dist * glm::length(ray.direction), isect.t);
    bool sampleMedium = t < isect.t;
    if (sampleMedium) {
        glm::vec3 samplePoint = ray.origin + t * ray.direction;
        mi->samplePoint = samplePoint;
        mi->wo = -ray.direction;
        mi->medium = mediumIndex;
    }

    // Compute transmittance and sample density
    glm::vec3 Tr = glm::exp(-medium.sigma_t * glm::min(t, MAX_FLOAT) * glm::length(ray.direction));
    
    // Return scattering weighting factor
    glm::vec3 density = sampleMedium ? (medium.sigma_t * Tr) : Tr;
  
    // TODO: change this to account for pdfs of other spectral wavelengths...
    float pdf = density[0];

    return sampleMedium ? (Tr * medium.sigma_s / pdf) : (Tr / pdf);
}

inline __host__ __device__
glm::vec3 Sample_heterogeneous(
    const Medium& medium,
    PathSegment& segment,
    const ShadeableIntersection& isect,
    MediumInteraction* mi,
    const nanovdb::NanoGrid<float>* media_density,
    int mediumIndex,
    GuiParameters& gui_params,
    thrust::default_random_engine& rng, 
    thrust::uniform_real_distribution<float>& u01)
{
    // Transform ray to local medium space
    Ray worldRay = segment.ray;

    Ray localRay; 
    localRay.origin = glm::vec3(medium.worldToMedium * glm::vec4(worldRay.origin, 1.0f));
    localRay.direction = glm::vec3(medium.worldToMedium * glm::vec4(worldRay.direction, 0.0f));
    localRay.direction_inv = 1.0f / localRay.direction;
    float rayTMax = isect.t * glm::length(worldRay.direction);

    // Compute tmin and tmax of ray overlap with medium bounds
    glm::vec3 localBBMin = glm::vec3(0.0f);
    glm::vec3 localBBMax = glm::vec3(1.0f);
    float tMin, tMax, t;
    if (!aabbIntersectionTest(segment, localBBMin, localBBMax, localRay, tMin, tMax, t, false)) {
        return glm::vec3(1.0f);
    }

    // Run delta tracking to sample medium interaction
    t = glm::max(tMin, 0.0f);
    glm::vec3 samplePoint = localRay.origin + t * localRay.direction;
    while (true) {
        t = -glm::log(1.0f - u01(rng)) * medium.invMaxDensity / getSigmaT(medium, gui_params)[0]; // TODO: sigma_t is a float for heterogeneous medium
        if (t >= tMax) {
            break;
        }

        samplePoint = localRay.origin + t * localRay.direction;
        if (Density_heterogeneous(medium, media_density, samplePoint, segment) * medium.invMaxDensity > u01(rng)) {
            mi->samplePoint = worldRay.origin + t * worldRay.direction;
            mi->wo = -segment.ray.direction;
            mi->medium = mediumIndex;
            return gui_params.sigma_s / getSigmaT(medium, gui_params);
        }
        /*if (segment.remainingBounces > 0) {
            segment.remainingBounces--;
        }
        if (segment.remainingBounces == 0) {
            return glm::vec3(1.0f);
        }*/
    }

    return glm::vec3(1.0f);
}

inline __host__ __device__ bool IsMediumTransition(const MediumInterface& mi)
{ 
    return mi.inside != mi.outside; 
}

inline __host__ __device__
void Sample_Li(
    const Geom& light,
    const glm::vec3& intersect_point,
    glm::vec3& wi,
    float& pdf_L,
    thrust::default_random_engine& rng,
    thrust::uniform_real_distribution<float>& u01) 
{
    float absDot = 0.0f;
    if (light.type == SQUAREPLANE) {
        glm::vec2 p_obj_space = glm::vec2(u01(rng) - 0.5f, u01(rng) - 0.5f);
        glm::vec3 p_world_space = glm::vec3(light.transform * glm::vec4(p_obj_space.x, p_obj_space.y, 0.0f, 1.0f));
        wi = glm::normalize(glm::vec3(p_world_space - intersect_point));
        absDot = glm::dot(wi, glm::normalize(glm::vec3(light.invTranspose * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f))));

        if (absDot < 0.0001f) {
            absDot = glm::abs(absDot);
            // pdf of square plane light = distanceSq / (absDot * lightArea)
            float dist = glm::length(p_world_space - intersect_point);
            if (absDot > 0.0001f) {
                pdf_L = (dist * dist) / (absDot * light.scale.x * light.scale.y);
            }
        }
        else {
            pdf_L = 0.0f;
        }
    }
}

// function to randomly choose a light, randomly choose point on light, compute LTE with that random point, and generate ray for shadow casting
inline __host__ __device__
glm::vec3 computeDirectLightSamplePreVis(
    int idx,
    PathSegment* pathSegments,
    Material& material,
    Material* materials,
    ShadeableIntersection &intersection,
    Medium* media,
    MISLightRay* direct_light_rays,
    MISLightIntersection* direct_light_isects,
    Light* lights,
    int num_lights,
    Geom* geoms,
    GuiParameters& gui_params,
    thrust::default_random_engine& rng, 
    thrust::uniform_real_distribution<float>& u01) {

    ////////////////////////////////////////////////////
    // LIGHT SAMPLED
    ////////////////////////////////////////////////////

    glm::vec3 intersect_point = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;
    if (intersection.mi.medium >= 0) {
        intersect_point = intersection.mi.samplePoint;
    }

    // choose light to directly sample
    direct_light_rays[idx].light_ID = lights[glm::min((int)(glm::floor(u01(rng) * (float)num_lights)), num_lights - 1)].geom_ID;

    Geom& light = geoms[direct_light_rays[idx].light_ID];

    Material& light_material = materials[light.materialid];

    // generate light sampled wi
    glm::vec3 wi = glm::vec3(0.0f);
    float absDot = 0.0f;
    glm::vec3 f = glm::vec3(0.0f);
    float pdf_L = 0.0f;

    if (light.type == SQUAREPLANE) {
        glm::vec2 p_obj_space = glm::vec2(u01(rng) - 0.5f, u01(rng) - 0.5f);
        glm::vec3 p_world_space = glm::vec3(light.transform * glm::vec4(p_obj_space.x, p_obj_space.y, 0.0f, 1.0f));
        wi = glm::normalize(glm::vec3(p_world_space - intersect_point));
        absDot = glm::dot(wi, glm::normalize(glm::vec3(light.invTranspose * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f))));

        if (absDot < 0.0001f) {
            absDot = glm::abs(absDot);
            // pdf of square plane light = distanceSq / (absDot * lightArea)
            float dist = glm::length(p_world_space - intersect_point);
            if (absDot > 0.0001f) {
                pdf_L = (dist * dist) / (absDot * light.scale.x * light.scale.y);
            }
        }
        else {
            pdf_L = 0.0f;
        }
    }

    direct_light_rays[idx].ray.origin = intersect_point + (wi * 0.001f);
    direct_light_rays[idx].ray.direction = wi;
    direct_light_rays[idx].ray.direction_inv = 1.0f / wi;
    direct_light_rays[idx].medium = pathSegments[idx].medium;


    // SURFACE INTERACTION
    if (intersection.mi.medium == -1) {
        absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
        // generate f, pdf, absdot from light sampled wi
        if (material.type == SPEC_BRDF) {
            // spec refl
            direct_light_rays[idx].f = glm::vec3(0.0f);
        }
        else if (material.type == SPEC_BTDF) {
            // spec refr
            direct_light_rays[idx].f = glm::vec3(0.0f);
        }
        else if (material.type == SPEC_GLASS) {
            // spec glass
            direct_light_rays[idx].f = glm::vec3(0.0f);
        }
        else {
            direct_light_rays[idx].f = material.R * 0.31831f; // INV_PI
        }

        // LTE = f * Li * absDot / pdf
        if (pdf_L <= 0.0001f) {
            direct_light_isects[idx].LTE = glm::vec3(0.0f, 0.0f, 0.0f);
        }
        else {
            direct_light_isects[idx].LTE = (float)num_lights * light_material.emittance * light_material.R * direct_light_rays[idx].f * absDot / pdf_L;
        }
    }
    // VOLUME INTERACTION
    else {
        float p = evaluatePhaseHG(intersection.mi.wo, wi, media[intersection.mi.medium].g, gui_params.g);
        direct_light_rays[idx].f = glm::vec3(p);
        if (pdf_L <= 0.0001f) {
            direct_light_isects[idx].LTE = glm::vec3(0.0f, 0.0f, 0.0f);
        }
        else {
            direct_light_isects[idx].LTE = (float)num_lights * light_material.emittance * light_material.R * direct_light_rays[idx].f / pdf_L;
        }
    }
}

inline __host__ __device__
glm::vec3 getMajorant(const Medium& medium, GuiParameters& gui_params)
{
    return ((medium.maxDensity + gui_params.density_offset) * gui_params.density_scale) * (gui_params.sigma_a + gui_params.sigma_s);
}

inline __host__ __device__
void getCoefficients(
    const nanovdb::NanoGrid<float>* media_density,
    GuiParameters& gui_params,
    const Medium& medium,
    const glm::vec3& samplePoint,
    PathSegment& segment,
    glm::vec3& scattering,
    glm::vec3& absorption,
    glm::vec3& null)
{
    glm::vec3 localSamplePoint = glm::vec3(medium.worldToMedium * glm::vec4(samplePoint, 1.0));
    float density;
    if (medium.type == HOMOGENEOUS) {
        density = Density_homogeneous(medium);
    }
    else {
        density = Density_heterogeneous(medium, media_density, localSamplePoint, segment);
    }
    density = (density + gui_params.density_offset) * gui_params.density_scale;
    scattering = density * glm::vec3(gui_params.sigma_s);
    absorption = density * glm::vec3(gui_params.sigma_a);
    null = getMajorant(medium, gui_params) - (scattering + absorption);
}

// This returns Tr
inline __host__ __device__
glm::vec3 Sample_channel_direct(
    int idx,
    const Medium& medium,
    PathSegment& segment,
    const MISLightIntersection& isect,
    const nanovdb::NanoGrid<float>* media_density,
    MISLightRay& direct_ray,
    float& ray_t,
    glm::vec3& T_ray,
    GuiParameters& gui_params,
    thrust::default_random_engine& rng,
    thrust::uniform_real_distribution<float>& u01)
{
    Ray worldRay = direct_ray.ray;

    Ray localRay;
    localRay.origin = glm::vec3(medium.worldToMedium * glm::vec4(worldRay.origin, 1.0f));
    localRay.direction = glm::vec3(medium.worldToMedium * glm::vec4(worldRay.direction, 0.0f));
    localRay.direction_inv = 1.0f / localRay.direction;

    // Compute tmin and tmax of ray overlap with medium bounds
    glm::vec3 localBBMin = glm::vec3(0.0f);
    glm::vec3 localBBMax = glm::vec3(1.0f);
    float tMin, tMax, t;
    if (!aabbIntersectionTest(segment, localBBMin, localBBMax, localRay, tMin, tMax, t, false)) {
        return glm::vec3(1.0f);
    }

    glm::vec3 T_maj = glm::vec3(1.0f);
    int channel = segment.rgbWavelength;
    tMin = glm::max(tMin, 0.0f);

    while (true) {
        t = tMin - glm::log(1.0f - u01(rng)) / getMajorant(medium, gui_params)[channel];
        bool sampleMedium = t < tMax;
        t = glm::min(t, tMax);

        if (sampleMedium) {
            ray_t = t;
            glm::vec3 samplePoint = worldRay.origin + t * worldRay.direction;

            T_maj *= glm::exp(-getMajorant(medium, gui_params) * (t - tMin));
            glm::vec3 scattering, absorption, null;
            getCoefficients(media_density, gui_params, medium, samplePoint, segment, scattering, absorption, null);
            glm::vec3 majorant = getMajorant(medium, gui_params);

            float pdf = T_maj[channel] * majorant[channel];
            //if (pdf < EPSILON) {
            //    return glm::vec3(0.0f);
            //}
            T_ray *= T_maj * null / pdf;
            direct_ray.r_l *= T_maj * majorant / pdf;
            direct_ray.r_u *= T_maj * null / pdf;

            glm::vec3 Tr = T_ray / (direct_ray.r_l + direct_ray.r_u);
            if (glm::max(Tr.x, glm::max(Tr.y, Tr.z)) < 0.05f) {
                float q = 0.75f;
                if (u01(rng) < q)
                    T_ray = glm::vec3(0.0f);
                else
                    T_ray /= 1.0f - q;
            }

            if (glm::length(T_ray) < 0.0001f) {
                return glm::vec3(1.0f);
            }

            tMin = t;
            T_maj = glm::vec3(1.0f);
        }
        else {
            T_maj *= glm::exp(-getMajorant(medium, gui_params) * (t - tMin));
            if (glm::length(T_maj) < 0.0001f) {
                return glm::vec3(1.0f);
            }

            return T_maj;
        }
    }
}

// determines if sample point is occluded by scene geometry
inline __host__ __device__
glm::vec3 computeVisibility(
    int idx,
    PathSegment& segment,
    Geom* geoms,
    Tri* tris,
    Medium* media,
    const nanovdb::NanoGrid<float>* media_density,
    MISLightRay& direct_ray,
    MISLightIntersection& direct_isect,
    Light* lights,
    LBVHNode* lbvh,
    GuiParameters& gui_params,
    SceneInfo& scene_info,
    thrust::default_random_engine& rng,
    thrust::uniform_real_distribution<float>& u01)
{
    glm::vec3 T_ray = glm::vec3(1.0f);

    int num_iters = 0;

    while (true) {
        // Surface Intersection
        float t_min = MAX_INTERSECT_DIST;
        int obj_ID = -1;
        float t;
        float tMin, tMax;
        glm::vec3 tmp_normal;
        int mat_id = -1;

        for (int i = 0; i < scene_info.geoms_size; ++i)
        {
            Geom& geom = geoms[i];

            if (geom.type == SPHERE) {
                t = sphereIntersectionTest(geom, direct_ray.ray, tmp_normal);
            }
            else if (geom.type == SQUAREPLANE) {
                t = squareplaneIntersectionTest(geom, direct_ray.ray, tmp_normal);
            }
            else if (geom.type == CUBE) {
                t = boxIntersectionTest(geom, direct_ray.ray, tmp_normal);
            }
            else if (geom.type == MESH) {
                t = lbvhIntersectionTest(segment, lbvh, tris, direct_ray.ray, geom.triangleCount, tmp_normal, false);
            }

            if (t_min > t)
            {
                t_min = t;
                obj_ID = i;
                mat_id = geom.materialid;

                // Check if surface is medium transition
                if (IsMediumTransition(geom.mediumInterface)) {
                    direct_isect.mediumInterface = geom.mediumInterface;
                }
                else {
                    direct_isect.mediumInterface.inside = direct_ray.medium;
                    direct_isect.mediumInterface.outside = direct_ray.medium;
                }
            }
        }

        if (scene_info.media_size > 0) {
            for (int j = 0; j < scene_info.media_size; j++) {
                if (media[j].type == HOMOGENEOUS) continue;

                const Medium& medium = media[j];
                bool intersectAABB = aabbIntersectionTest(segment, medium.aabb_min, medium.aabb_max, direct_ray.ray, tMin, tMax, t, false);

                if (intersectAABB && t_min > t) {
                    t_min = t;
                    obj_ID = -2;
                    mat_id = -1;

                    // TODO: change this to handle more advanced cases
                    direct_isect.mediumInterface.inside = j;
                    direct_isect.mediumInterface.outside = -1;
                }
            }
        }
        
        // if we did not intersect an object or intersected object is not a "invisible" bounding box, the ray is occluded
        if (obj_ID == -1 || (obj_ID != -1 && obj_ID != direct_ray.light_ID && mat_id != -1)) {
            num_iters++;
            return glm::vec3(0.0f);
        }

        // if the current ray has a medium, then attenuate throughput based on transmission and distance traveled
        if (direct_ray.medium != -1) {
            if (media[direct_ray.medium].type == HOMOGENEOUS) {
                T_ray *= Tr_homogeneous(media[direct_ray.medium], direct_ray.ray, t_min);
            }
            else {
                glm::vec3 T_maj = Sample_channel_direct(idx, media[direct_ray.medium], segment, direct_isect, media_density, direct_ray, t_min, T_ray, gui_params, rng, u01);
                T_ray *= T_maj / T_maj[segment.rgbWavelength];
                direct_ray.r_l *= T_maj / T_maj[segment.rgbWavelength];
                direct_ray.r_u *= T_maj / T_maj[segment.rgbWavelength];
            }
        }

        if (glm::length(T_ray) < EPSILON) {
            return glm::vec3(0.0f);
        }

        // if the intersected object IS the light source we selected, we are done
        if (obj_ID == direct_ray.light_ID) {
            num_iters++;
            return T_ray;
        }

        num_iters++;
        // We encountered a bounding box/entry/exit of a volume, so we must change our medium value, update the origin, and traverse again
        direct_ray.ray.origin = direct_ray.ray.origin + (direct_ray.ray.direction * (t_min + 0.001f));

        // TODO: generalize to support both homogeneous and heterogeneous volumes
        /*r.medium = glm::dot(r.ray.direction, tmp_normal) > 0 ? isect.mediumInterface.outside :
            isect.mediumInterface.inside;*/
        direct_ray.medium = insideMedium(segment, tMin, tMax, num_iters) ? direct_isect.mediumInterface.inside : direct_isect.mediumInterface.outside;
    }
}

// function to randomly choose a light, randomly choose point on light, compute LTE with that random point, and compute visibility
inline __host__ __device__
glm::vec3 directLightSample(
    int idx,
    bool is_medium,
    PathSegment& segment,
    Material* materials,
    ShadeableIntersection& intersection,
    Geom* geoms,
    Tri* tris,
    Medium* media,
    const nanovdb::NanoGrid<float>* media_density,
    MISLightRay& direct_ray,
    MISLightIntersection& direct_isect,
    Light* lights,
    LBVHNode* lbvh,
    GuiParameters& gui_params,
    SceneInfo& scene_info,
    thrust::default_random_engine& rng,
    thrust::uniform_real_distribution<float>& u01) 
{
    // calculate point on surface or medium from which the light ray should originate
    glm::vec3 intersect_point = segment.ray.origin + intersection.t * segment.ray.direction;
    if (intersection.mi.medium >= 0) {
        intersect_point = intersection.mi.samplePoint;
    }

    // choose light to directly sample
    direct_ray.light_ID = lights[glm::min((int)(glm::floor(u01(rng) * (float)scene_info.lights_size)), scene_info.lights_size - 1)].geom_ID;
    Geom& light = geoms[direct_ray.light_ID];
    Material& light_material = materials[light.materialid];

    // get light wi direction and pdf
    glm::vec3 wi = glm::vec3(0.0f);
    float pdf_L = 0.0f;
    Sample_Li(light, intersect_point, wi, pdf_L, rng, u01);

    // store direct light ray information for visibility testing
    direct_ray.ray.origin = intersect_point + (wi * 0.001f);
    direct_ray.ray.direction = wi;
    direct_ray.ray.direction_inv = 1.0f / wi;
    direct_ray.medium = segment.medium;
    direct_ray.r_l = glm::vec3(1.0f);
    direct_ray.r_u = glm::vec3(1.0f);

    float pdf_B = 0.0f;

    // Surface interaction
    if (!is_medium) {
        Material& material = materials[intersection.materialId];
        float absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));

        pdf_B = absDot * 0.31831f;
        direct_ray.f = material.R * 0.31831f; // INV_PI

        // LTE = f * Li * absDot / pdf
        if (pdf_L < 0.00001f) {
            direct_isect.LTE = glm::vec3(0.0f, 0.0f, 0.0f);
            return glm::vec3(0.0f);
        }
        else if (pdf_B < 0.00001f) {
            direct_isect.LTE = glm::vec3(0.0f, 0.0f, 0.0f);
            return glm::vec3(0.0f);
        }
        else {
            direct_isect.LTE = light_material.emittance * light_material.R * direct_ray.f * absDot;
        }
    }
    // Medium interaction
    else {
        // evaluate phase function for light sample direction
        float p = evaluatePhaseHG(intersection.mi.wo, wi, media[intersection.mi.medium].g, gui_params.g);
        pdf_B = p;

        if (pdf_B < EPSILON) {
            direct_isect.LTE = glm::vec3(0.0f, 0.0f, 0.0f);
            return glm::vec3(0.0f);
        }

        direct_ray.f = glm::vec3(p);

        if (pdf_L < EPSILON) {
            direct_isect.LTE = glm::vec3(0.0f, 0.0f, 0.0f);
            return glm::vec3(0.0f);
        }
        else {
            direct_isect.LTE = light_material.emittance * light_material.R * direct_ray.f;
        }
    }

    // compute visibility
    glm::vec3 T_ray = computeVisibility(idx, segment, geoms, tris, media, media_density,
        direct_ray, direct_isect, lights, lbvh, gui_params, scene_info, rng, u01);
 
    if (gui_params.importance_sampling == NEE) {
        direct_isect.LTE *= T_ray * (float)scene_info.lights_size / pdf_L;
    }
    else if (gui_params.importance_sampling == UNI_NEE_MIS) {
        direct_ray.r_l *= segment.r_u * pdf_L / (float)scene_info.lights_size;
        direct_ray.r_u *= segment.r_u * pdf_B;

        direct_isect.LTE *= T_ray / (direct_ray.r_l + direct_ray.r_u);
    }
    return direct_isect.LTE;
}

inline __host__ __device__
MediumEvent sampleMediumEvent(float pAbsorb, float pScatter, float pNull, float rng_val) {
    if (pAbsorb > rng_val) {
        return ABSORB;
    }

    if (pScatter + pAbsorb > rng_val) {
        return REAL_SCATTER;
    }

    return NULL_SCATTER;
}

// This returns Tr
inline __host__ __device__
glm::vec3 Sample_channel(
    int path_index,
    int mediumIndex,
    PathSegment& segment,
    ShadeableIntersection& isect,
    MISLightRay& direct_light_ray,
    MISLightIntersection& direct_light_isect,
    Geom* geoms,
    Tri* tris,
    Light* lights,
    Medium* media,
    Material* materials,
    MediumInteraction* mi,
    LBVHNode* lbvh,
    const nanovdb::NanoGrid<float>* media_density,
    GuiParameters& gui_params,
    SceneInfo& scene_info,
    thrust::default_random_engine& rng,
    thrust::uniform_real_distribution<float>& u01,
    bool& scattered)
{
    Ray worldRay = segment.ray;

    Ray localRay;
    localRay.origin = glm::vec3(media[mediumIndex].worldToMedium * glm::vec4(worldRay.origin, 1.0f));
    localRay.direction = glm::vec3(media[mediumIndex].worldToMedium * glm::vec4(worldRay.direction, 0.0f));
    localRay.direction_inv = 1.0f / localRay.direction;

    // Compute tmin and tmax of ray overlap with medium bounds
    glm::vec3 localBBMin = glm::vec3(0.0f);
    glm::vec3 localBBMax = glm::vec3(1.0f);
    float tMin, tMax, t;
    if (!aabbIntersectionTest(segment, localBBMin, localBBMax, localRay, tMin, tMax, t, false)) {
        return glm::vec3(1.0f);
    }

    glm::vec3 T_maj = glm::vec3(1.0f);
    int channel = segment.rgbWavelength;
    tMin = glm::max(tMin, 0.0f);

    while (true) {
        t = tMin - glm::log(1.0f - u01(rng)) / getMajorant(media[mediumIndex], gui_params)[channel];
        bool sampleMedium = t < tMax;
        t = glm::min(t, tMax);

        if (sampleMedium) {
            glm::vec3 samplePoint = worldRay.origin + t * worldRay.direction;

            T_maj *= glm::exp(-getMajorant(media[mediumIndex], gui_params) * (t - tMin));

            // Set medium properties
            mi->samplePoint = samplePoint;
            mi->wo = -worldRay.direction;
            mi->medium = mediumIndex;
            isect.mi = *mi;

            // Handle medium interaction
            //int heroChannel = 0;

            glm::vec3 scattering, absorption, null;
            getCoefficients(media_density, gui_params, media[mi->medium], mi->samplePoint, segment, scattering, absorption, null);

            glm::vec3 majorant = getMajorant(media[mi->medium], gui_params);
            float pAbsorb = absorption[channel] / majorant[channel];
            float pScatter = scattering[channel] / majorant[channel];
            float pNull = 1.0f - pAbsorb - pScatter;

            // choose a medium event to sample (absorption, real scattering, or null scattering
            MediumEvent eventType = sampleMediumEvent(pAbsorb, pScatter, pNull, u01(rng));

            if (eventType == ABSORB) {
                segment.remainingBounces = 0;
                return glm::vec3(1.0f);
            }
            else if (eventType == NULL_SCATTER) {
                float pdf = T_maj[channel] * null[channel];
                if (pdf < EPSILON) {
                    segment.rayThroughput = glm::vec3(0.0f);
                    return glm::vec3(1.0f);
                }
                else {
                    segment.rayThroughput *= T_maj * null / pdf;
                    segment.r_u *= T_maj * null / pdf;
                    segment.r_l *= T_maj * majorant / pdf;
                    //pathSegments[path_index].ray.origin = mi->samplePoint;
                }

                if (glm::length(segment.rayThroughput) <= 0.00001f || glm::length(segment.r_u) <= 0.00001f) {
                    return glm::vec3(1.0f);
                }
            }
            else {
                // Stop after reaching max depth
                if (segment.remainingBounces <= 0) {
                    segment.remainingBounces = 0;
                    return glm::vec3(1.0f);
                }



                float pdf = T_maj[channel] * scattering[channel];
                if (pdf < EPSILON) {
                    segment.remainingBounces = 0;
                    return glm::vec3(1.0f);
                }
                segment.rayThroughput *= T_maj * scattering / pdf;
                segment.r_u *= T_maj * scattering / pdf;

                bool sampleLight = (glm::length(segment.rayThroughput) > EPSILON && glm::length(segment.r_u) > EPSILON);
                if (sampleLight) {
                    // Direct light sampling
                    if (gui_params.importance_sampling == NEE || gui_params.importance_sampling == UNI_NEE_MIS) {
                        // Direct light sampling
                        glm::vec3 Ld = directLightSample(path_index, true, segment, materials, isect, geoms, tris,
                          media, media_density, direct_light_ray, direct_light_isect, lights, lbvh, gui_params, scene_info, rng, u01);

                        segment.accumulatedIrradiance += segment.rayThroughput * Ld;
                    }

                    // Sample phase function
                    glm::vec3 phase_wi;
                    float phase_pdf = 0.f;
                    float phase_p = Sample_p(-segment.ray.direction, &phase_wi, &phase_pdf, glm::vec2(u01(rng), u01(rng)), media[segment.medium].g, gui_params.g);
                    if (phase_pdf < EPSILON) {
                        segment.remainingBounces = 0;
                        return glm::vec3(1.0f);
                    }

                    // Update ray segment
                    segment.rayThroughput *= phase_p / phase_pdf;
                    segment.r_l = segment.r_u / phase_pdf;
                    segment.ray.direction = phase_wi;
                    segment.ray.direction_inv = 1.0f / phase_wi;
                    segment.ray.origin = mi->samplePoint + phase_wi * 0.001f;
                    segment.medium = mi->medium;
                    segment.remainingBounces--;
                    scattered = true;
                }
                else {
                    segment.remainingBounces = 0;
                }
                return glm::vec3(1.0f);
            }

            tMin = t;
            T_maj = glm::vec3(1.0f);
        }
        else {
            // Set medium properties
            mi->medium = -1;
            isect.mi = *mi;
            T_maj *= glm::exp(-getMajorant(media[mediumIndex], gui_params) * (t - tMin));
            if (glm::length(T_maj) < 0.0001f) {
                return glm::vec3(1.0f);
            }

            return T_maj;
        }
    }
}

//inline __host__ __device__
//bool handleMediumInteraction(
//    int idx,
//    int max_depth,
//    const glm::vec3 T_maj, 
//    PathSegment* pathSegments,
//    Material* materials,
//    ShadeableIntersection& isect,
//    const MediumInteraction& mi,
//    Geom* geoms,
//    int geoms_size,
//    Tri* tris,
//    Medium* media,
//    int media_size,
//    const nanovdb::NanoGrid<float>* media_density,
//    MISLightRay* direct_light_rays,
//    MISLightIntersection* direct_light_isects,
//    Light* lights,
//    int num_lights,
//    LBVHNode* lbvh,
//    GuiParameters& gui_params,
//    thrust::default_random_engine& rng,
//    thrust::uniform_real_distribution<float>& u01)
//{
//    int heroChannel = 0;
//
//    glm::vec3 scattering, absorption, null;
//    getCoefficients(media_density, gui_params, media[mi.medium], mi.samplePoint, pathSegments[idx], scattering, absorption, null);
//    
//    glm::vec3 majorant = getMajorant(media[mi.medium], gui_params);
//    float pAbsorb = absorption[heroChannel] / majorant[heroChannel];
//    float pScatter = scattering[heroChannel] / majorant[heroChannel];
//    float pNull = 1.0f - pAbsorb - pScatter;
//
//    // choose a medium event to sample (absorption, real scattering, or null scattering
//    MediumEvent eventType = sampleMediumEvent(pAbsorb, pScatter, pNull, u01(rng));
//
//    if (eventType == ABSORB) {
//        pathSegments[idx].remainingBounces = 0;
//        return false;
//    }
//    else if (eventType == NULL_SCATTER) {
//        // TODO: maybe decrement remaining bounces
//        pathSegments[idx].prev_event_was_real = false;
//        float pdf = T_maj[heroChannel] * null[heroChannel];
//        if (pdf < EPSILON) {
//            pathSegments[idx].rayThroughput = glm::vec3(0.0f);
//            return false;
//        }
//        else {
//            pathSegments[idx].rayThroughput *= T_maj * null / pdf;
//            pathSegments[idx].r_u *= T_maj * null / pdf;
//            pathSegments[idx].r_l *= T_maj * majorant / pdf;
//            pathSegments[idx].ray.origin = mi.samplePoint;
//        }
//
//        if (glm::length(pathSegments[idx].rayThroughput) <= 0.00001f || glm::length(pathSegments[idx].r_u) <= 0.00001f) {
//            pathSegments[idx].remainingBounces = 0;
//            return false;
//        }
//        else {
//            //pathSegments[idx].remainingBounces--;
//        }
//        return true;
//    }
//    else {
//        pathSegments[idx].prev_event_was_real = true;
//        pathSegments[idx].realPathLength++;
//
//        // Stop after reaching max depth
//        if (pathSegments[idx].remainingBounces <= 0) {
//            pathSegments[idx].remainingBounces = 0;
//            return false;
//        }
//
//        float pdf = T_maj[heroChannel] * scattering[heroChannel];
//        if (pdf < EPSILON) {
//            pathSegments[idx].remainingBounces = 0;
//            return false;
//        }
//        pathSegments[idx].rayThroughput *= T_maj * scattering / pdf;
//        pathSegments[idx].r_u *= T_maj * scattering / pdf;
//
//        bool sampleLight = (glm::length(pathSegments[idx].rayThroughput) > EPSILON || glm::length(pathSegments[idx].r_u) > EPSILON);
//        if (sampleLight) {
//            // Direct light sampling
//            if (gui_params.importance_sampling == NEE || gui_params.importance_sampling == UNI_NEE_MIS) {
//                glm::vec3 Ld = directLightSample(idx, true, pathSegments, materials, isect, geoms, geoms_size, tris,
//                  media, media_size, media_density, direct_light_rays, direct_light_isects, lights, num_lights, lbvh, gui_params, rng, u01);
//                  
//                pathSegments[idx].accumulatedIrradiance += pathSegments[idx].rayThroughput * Ld;
//
//            }
//
//            // Sample phase function
//            glm::vec3 phase_wi;
//            float phase_pdf = 0.f;
//            float phase_p = Sample_p(-pathSegments[idx].ray.direction, &phase_wi, &phase_pdf, glm::vec2(u01(rng), u01(rng)), media[pathSegments[idx].medium].g, gui_params.g);
//            if (phase_pdf < EPSILON) {
//                pathSegments[idx].remainingBounces = 0;
//                return false;
//            }
//
//            // Update ray segment
//            pathSegments[idx].rayThroughput *= phase_p / phase_pdf;
//            pathSegments[idx].r_l = pathSegments[idx].r_u / phase_pdf;
//            pathSegments[idx].ray.direction = phase_wi;
//            pathSegments[idx].ray.direction_inv = 1.0f / phase_wi;
//            pathSegments[idx].ray.origin = mi.samplePoint;
//            pathSegments[idx].medium = mi.medium;
//            pathSegments[idx].remainingBounces--;
//
//            return true;
//        }
//        else {
//            pathSegments[idx].remainingBounces = 0;
//            return false;
//        }       
//    }
//}