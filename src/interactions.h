#pragma once

#include "intersections.h"
#include <thrust/random.h>

#include "../external/include/openvdb/nanovdb/nanovdb/util//CudaDeviceBuffer.h"


//#define USE_SCHLICK_APPROX

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

inline __host__ __device__
float evaluatePhaseHG(const glm::vec3& wo, const glm::vec3& wi, float g)
{
    float cosTheta = glm::dot(wo, wi);
    float denom = 1 + g * g + 2 * g * cosTheta;
    return INV_FOUR_PI * (1 - g * g) / (denom * glm::sqrt(denom));
}

inline __host__ __device__
float Sample_p(const glm::vec3& wo, glm::vec3* wi, const glm::vec2& u, float g)
{
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

    return evaluatePhaseHG(wo, *wi, g);
}

inline __host__ __device__ float D_heterogeneous(const Medium& medium, const nanovdb::NanoGrid<float>* media_density, glm::vec3 sample_index)
{
    // read density value from the grid
    glm::vec3 min_cell_index = glm::vec3(0, 0, 0);
    glm::vec3 max_cell_index = glm::vec3(medium.gx, medium.gy, medium.gz);
    if (sample_index.x < min_cell_index.x || sample_index.x >= max_cell_index.x ||
        sample_index.y < min_cell_index.y || sample_index.y >= max_cell_index.y ||
        sample_index.z < min_cell_index.z || sample_index.z >= max_cell_index.z)
        return 0;
    auto gpuAcc = media_density->getAccessor();
    auto density = gpuAcc.getValue(nanovdb::Coord(sample_index.x, sample_index.y, sample_index.z));
    /*if (glm::abs(density) < 0.001f) {
        density = 0.0f;
    }*/
    return (density >= 0.0f) ? density : -density;
}

inline __host__ __device__ float Density_heterogeneous(const Medium& medium, const nanovdb::NanoGrid<float>* media_density, glm::vec3 sample_point)
{
    // find the sample point's voxel
    glm::vec3 pSamples(sample_point.x * medium.gx - 0.5f, sample_point.y * medium.gy - 0.5f, sample_point.z * medium.gz - 0.5f);
    glm::vec3 pi = glm::floor(pSamples);
    glm::vec3 d = pSamples - pi;
    //return D_heterogeneous(medium, media_density, pi);

    // trilinear sampling of density values nearest to sampling point
    float d00 = glm::mix(D_heterogeneous(medium, media_density, pi), D_heterogeneous(medium, media_density, pi + glm::vec3(1, 0, 0)), d.x);
    float d10 = glm::mix(D_heterogeneous(medium, media_density, pi + glm::vec3(0, 1, 0)), D_heterogeneous(medium, media_density, pi + glm::vec3(1, 1, 0)), d.x);
    float d01 = glm::mix(D_heterogeneous(medium, media_density, pi + glm::vec3(0, 0, 1)), D_heterogeneous(medium, media_density, pi + glm::vec3(1, 0, 1)), d.x);
    float d11 = glm::mix(D_heterogeneous(medium, media_density, pi + glm::vec3(0, 1, 1)), D_heterogeneous(medium, media_density, pi + glm::vec3(1, 1, 1)), d.x);
    float d0 = glm::mix(d00, d10, d.y);
    float d1 = glm::mix(d01, d11, d.y);
    return glm::mix(d0, d1, d.z);
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
    t = tMin;
    glm::vec3 samplePoint = localRay.origin + t * localRay.direction;
    while (true) {
        t -= glm::log(1.0f - u01(rng)) * medium.invMaxDensity / medium.sigma_t[0];  // TODO: sigma_t is a float for heterogeneous medium
        if (t >= tMax) {
            break;
        }

        samplePoint = localRay.origin + t * localRay.direction;
        float density = Density_heterogeneous(medium, media_density, samplePoint);
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
    // QUESTION: is pdf calculation correct?
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
    //segment.accumulatedIrradiance += glm::vec3(100, 0, 0);

    // Run delta tracking to sample medium interaction
    t = tMin;
    glm::vec3 samplePoint = localRay.origin + t * localRay.direction;
    while (true) {
        t = -glm::log(1.0f - u01(rng)) * medium.invMaxDensity / medium.sigma_t[0]; // TODO: sigma_t is a float for heterogeneous medium
        if (t >= tMax) {
            break;
        }

        samplePoint = localRay.origin + t * localRay.direction;
        if (Density_heterogeneous(medium, media_density, samplePoint) * medium.invMaxDensity > u01(rng)) {
            mi->samplePoint = worldRay.origin + t * worldRay.direction;
            mi->wo = -segment.ray.direction;
            mi->medium = mediumIndex;
            return medium.sigma_s / medium.sigma_t;
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
        float p = evaluatePhaseHG(intersection.mi.wo, wi, media[intersection.mi.medium].g);
        direct_light_rays[idx].f = glm::vec3(p);
        if (pdf_L <= 0.0001f) {
            direct_light_isects[idx].LTE = glm::vec3(0.0f, 0.0f, 0.0f);
        }
        else {
            direct_light_isects[idx].LTE = (float)num_lights * light_material.emittance * light_material.R * direct_light_rays[idx].f / pdf_L;
        }
    }

}

// determines if sample point is occluded by scene geometry
inline __host__ __device__
glm::vec3 computeVisibility(
    int idx,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Tri* tris,
    int tris_size,
    Medium* media,
    int media_size,
    const nanovdb::NanoGrid<float>* media_density,
    MISLightRay* direct_light_rays,
    MISLightIntersection* direct_light_isects,
    Light* lights,
    int num_lights,
    BVHNode_GPU* bvh_nodes,
    thrust::default_random_engine& rng,
    thrust::uniform_real_distribution<float>& u01)
{
    MISLightRay r = direct_light_rays[idx];
    MISLightIntersection isect = direct_light_isects[idx];

    glm::vec3 Tr = glm::vec3(1.0f);

    int num_iters = 0;

    while (true) {
        // Surface Intersection
        float t_min = MAX_INTERSECT_DIST;
        int obj_ID = -1;
        float t;
        float tMin, tMax;
        glm::vec3 tmp_normal;
        int mat_id = -1;

        if (tris_size != 0) {
            int stack_pointer = 0;
            int cur_node_index = 0;
            int node_stack[32];
            BVHNode_GPU cur_node;
            glm::vec3 P;
            glm::vec3 s;
            float t1;
            float t2;
            float tmin;
            float tmax;
            while (true) {
                cur_node = bvh_nodes[cur_node_index];

                // (ray-aabb test node)
                t1 = (cur_node.AABB_min.x - r.ray.origin.x) * r.ray.direction_inv.x;
                t2 = (cur_node.AABB_max.x - r.ray.origin.x) * r.ray.direction_inv.x;

                tmin = glm::min(t1, t2);
                tmax = glm::max(t1, t2);

                t1 = (cur_node.AABB_min.y - r.ray.origin.y) * r.ray.direction_inv.y;
                t2 = (cur_node.AABB_max.y - r.ray.origin.y) * r.ray.direction_inv.y;

                tmin = glm::max(tmin, glm::min(t1, t2));
                tmax = glm::min(tmax, glm::max(t1, t2));

                t1 = (cur_node.AABB_min.z - r.ray.origin.z) * r.ray.direction_inv.z;
                t2 = (cur_node.AABB_max.z - r.ray.origin.z) * r.ray.direction_inv.z;

                tmin = glm::max(tmin, glm::min(t1, t2));
                tmax = glm::min(tmax, glm::max(t1, t2));

                if (tmax >= tmin) {
                    // we intersected AABB
                    if (cur_node.tri_index != -1) {
                        // this is leaf node
                        // triangle intersection test
                        Tri tri = tris[cur_node.tri_index];


                        t = glm::dot(tri.plane_normal, (tri.p0 - r.ray.origin)) / glm::dot(tri.plane_normal, r.ray.direction);
                        if (t >= -0.0001f) {
                            P = r.ray.origin + t * r.ray.direction;

                            // barycentric coords
                            s = glm::vec3(glm::length(glm::cross(P - tri.p1, P - tri.p2)),
                                glm::length(glm::cross(P - tri.p2, P - tri.p0)),
                                glm::length(glm::cross(P - tri.p0, P - tri.p1))) / tri.S;

                            if (s.x >= -0.0001f && s.x <= 1.0001f && s.y >= -0.0001f && s.y <= 1.0001f &&
                                s.z >= -0.0001f && s.z <= 1.0001f && (s.x + s.y + s.z <= 1.0001f) && (s.x + s.y + s.z >= -0.0001f) && t_min > t) {
                                t_min = t;
                                tmp_normal = glm::normalize(s.x * tri.n0 + s.y * tri.n1 + s.z * tri.n2);
                                mat_id = tri.mat_ID;
                                // Check if surface is medium transition
                                if (IsMediumTransition(tri.mediumInterface)) {
                                    isect.mediumInterface = tri.mediumInterface;
                                }
                                else {
                                    isect.mediumInterface.inside = r.medium;
                                    isect.mediumInterface.outside = r.medium;
                                }
                            }
                        }
                        // if last node in tree, we are done
                        if (stack_pointer == 0) {
                            break;
                        }
                        // otherwise need to check rest of the things in the stack
                        stack_pointer--;
                        cur_node_index = node_stack[stack_pointer];
                    }
                    else {
                        node_stack[stack_pointer] = cur_node.offset_to_second_child;
                        stack_pointer++;
                        cur_node_index++;
                    }
                }
                else {
                    // didn't intersect AABB, remove from stack
                    if (stack_pointer == 0) {
                        break;
                    }
                    stack_pointer--;
                    cur_node_index = node_stack[stack_pointer];
                }
            }
        }

        for (int i = 0; i < geoms_size; ++i)
        {
            Geom& geom = geoms[i];

            if (geom.type == SPHERE) {
                t = sphereIntersectionTest(geom, r.ray, tmp_normal);
            }
            else if (geom.type == SQUAREPLANE) {
                t = squareplaneIntersectionTest(geom, r.ray, tmp_normal);
            }
            else {
                t = boxIntersectionTest(geom, r.ray, tmp_normal);
            }

            if (t_min > t)
            {
                t_min = t;
                obj_ID = i;
                mat_id = geom.materialid;

                // Check if surface is medium transition
                if (IsMediumTransition(geom.mediumInterface)) {
                    isect.mediumInterface = geom.mediumInterface;
                }
                else {
                    isect.mediumInterface.inside = r.medium;
                    isect.mediumInterface.outside = r.medium;
                }
            }
        }

        for (int j = 0; j < media_size; j++) {
            if (media[j].type == HOMOGENEOUS) continue;

            const Medium& medium = media[j];
            bool intersectAABB = aabbIntersectionTest(pathSegments[idx], medium.aabb_min, medium.aabb_max, r.ray, tMin, tMax, t, false);

            if (intersectAABB && t_min > t) {
                t_min = t;
                obj_ID = -2;
                mat_id = -1;

                // TODO: change this to handle more advanced cases
                isect.mediumInterface.inside = j;
                isect.mediumInterface.outside = -1;
            }
        }

        // if we did not intersect an object or intersected object is not a "invisible" bounding box, the ray is occluded
        if (obj_ID == -1 || (obj_ID != -1 && obj_ID != r.light_ID && mat_id != -1)) {
            num_iters++;
            return glm::vec3(0.0f);
        }

        // if the current ray has a medium, then attenuate throughput based on transmission and distance traveled
        if (r.medium != -1) {
            if (media[r.medium].type == HOMOGENEOUS) {
                Tr *= Tr_homogeneous(media[r.medium], r.ray, t_min);
            }
            else {
                Tr *= Tr_heterogeneous(media[r.medium], pathSegments[idx], r, media_density, t_min, rng, u01);
            }
        }

        // if the intersected object IS the light source we selected, we are done
        if (obj_ID == r.light_ID) {
            num_iters++;
            return Tr;
        }

        num_iters++;
        // We encountered a bounding box/entry/exit of a volume, so we must change our medium value, update the origin, and traverse again
        glm::vec3 old_origin = r.ray.origin;
        r.ray.origin = old_origin + (r.ray.direction * (t_min + 0.01f));

        // TODO: generalize to support both homogeneous and heterogeneous volumes
        /*r.medium = glm::dot(r.ray.direction, tmp_normal) > 0 ? isect.mediumInterface.outside :
            isect.mediumInterface.inside;*/
        r.medium = insideMedium(pathSegments[idx], tMin, tMax, num_iters) ? isect.mediumInterface.inside : isect.mediumInterface.outside;
    }
}

// function to randomly choose a light, randomly choose point on light, compute LTE with that random point, and compute visibility
inline __host__ __device__
glm::vec3 directLightSample(
    int idx,
    PathSegment* pathSegments,
    Material* materials,
    ShadeableIntersection& intersection,
    Geom* geoms,
    int geoms_size,
    Tri* tris,
    int tris_size,
    Medium* media,
    int media_size,
    const nanovdb::NanoGrid<float>* media_density,
    MISLightRay* direct_light_rays,
    MISLightIntersection* direct_light_isects,
    Light* lights,
    int num_lights,
    BVHNode_GPU* bvh_nodes,
    thrust::default_random_engine& rng,
    thrust::uniform_real_distribution<float>& u01) 
{
    // calculate point on surface or medium from which the light ray should originate
    glm::vec3 intersect_point = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;
    if (intersection.mi.medium >= 0) {
        intersect_point = intersection.mi.samplePoint;
    }

    // choose light to directly sample
    direct_light_rays[idx].light_ID = lights[glm::min((int)(glm::floor(u01(rng) * (float)num_lights)), num_lights - 1)].geom_ID;
    Geom& light = geoms[direct_light_rays[idx].light_ID];
    Material& light_material = materials[light.materialid];

    // get light wi direction and pdf
    glm::vec3 wi = glm::vec3(0.0f);
    float pdf_L = 0.0f;
    Sample_Li(light, intersect_point, wi, pdf_L, rng, u01);

    // store direct light ray information for visibility testing
    direct_light_rays[idx].ray.origin = intersect_point + (wi * 0.001f);
    direct_light_rays[idx].ray.direction = wi;
    direct_light_rays[idx].ray.direction_inv = 1.0f / wi;
    direct_light_rays[idx].medium = pathSegments[idx].medium;

    // evaluate phase function
    float p = evaluatePhaseHG(intersection.mi.wo, wi, media[intersection.mi.medium].g);
    direct_light_rays[idx].f = glm::vec3(p);
    if (pdf_L <= 0.0001f) {
        direct_light_isects[idx].LTE = glm::vec3(0.0f, 0.0f, 0.0f);
    }
    else {
        direct_light_isects[idx].LTE = (float)num_lights * light_material.emittance * light_material.R * direct_light_rays[idx].f / pdf_L;
    }

    // compute visibility
    glm::vec3 visibilityTr = computeVisibility(idx, pathSegments, geoms, geoms_size, tris, tris_size, media, media_size, media_density,
        direct_light_rays, direct_light_isects, lights, num_lights, bvh_nodes, rng, u01);
    direct_light_isects[idx].LTE *= visibilityTr;

    return direct_light_isects[idx].LTE;
}

inline __host__ __device__
MediumEvent sampleMediumEvent(const Medium& medium, glm::vec4& sigma_maj, float rng_val) {

    float p_absorb = medium.sigma_a[0] / sigma_maj[0];
    if (p_absorb > rng_val) {
        return ABSORB;
    }

    float p_scatter = medium.sigma_s[0] / sigma_maj[0];
    if (p_scatter + p_absorb > rng_val) {
        return REAL_SCATTER;
    }
    
    return NULL_SCATTER;
}


inline __host__ __device__
glm::vec3 Sample_heterogeneous_FullVol(
    int max_depth,
    const Medium& medium,
    PathSegment& segment,
    const ShadeableIntersection& isect,
    MediumInteraction* mi,
    const nanovdb::NanoGrid<float>* media_density,
    int mediumIndex,
    thrust::default_random_engine& rng,
    thrust::uniform_real_distribution<float>& u01)
{
    return glm::vec3(0, 0, 0);
    /*Ray sampleRay = segment.ray;
    sampleRay.direction = glm::normalize(sampleRay.direction);
    sampleRay.direction_inv = 1.0f / glm::normalize(sampleRay.direction);
    float rayTMax = isect.t * glm::length(segment.ray.direction);


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




    glm::vec3 T_maj = glm::vec3(1.0f);
    bool done = false;



    bool scattered = false;
    bool terminated = false;
    glm::vec4 sigma_maj = glm::vec4(1);

    if (glm::length(segment.rayThroughput) <= 0.0f) {
        terminated = true;
        return glm::vec3(0.0);
    }

    // choose a medium event to sample (absorption, real scattering, or null scattering
    MediumEvent medium_event = sampleMediumEvent(medium, sigma_maj, u01(rng));
    if (medium_event == ABSORB) {
        terminated = true;
        return glm::vec3(0.0);
    }
    else if (medium_event == REAL_SCATTER) {
        if (segment.remainingBounces >= max_depth) {
            terminated = true;
            return glm::vec3(0.0);
        }
        segment.remainingBounces--;

        float pdf = T_maj[0] * mp.sigma_s[0];
        segment.rayThroughput *= T_maj * mp.sigma_s / pdf;
        r_u *= T_maj * mp.sigma_s / pdf;
    }*/
}