#include <cstdio>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define BLOCK_SIZE_1D 128
#define BLOCK_SIZE_2D 16

#define MIN_INTERSECT_DIST 0.0001f
#define MAX_INTERSECT_DIST 10000.0f

#define ENABLE_RECTS
#define ENABLE_SPHERES
#define ENABLE_TRIS
#define ENABLE_SQUAREPLANES


#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}



__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}



static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Tri* dev_tris = NULL;
static Light* dev_lights = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static LBVHNode* dev_lbvh = NULL;

static MISLightRay* dev_direct_light_rays = NULL;
static MISLightIntersection* dev_direct_light_isects = NULL;

static MISLightRay* dev_bsdf_light_rays = NULL;
static MISLightIntersection* dev_bsdf_light_isects = NULL;

static glm::vec3* dev_sample_colors = NULL;

int pixelcount;

void pathtraceInit(Scene* scene) {

	hst_scene = scene;

	const Camera& cam = hst_scene->state.camera;
	pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	/*cudaMalloc(&dev_tris, scene->num_tris * sizeof(Tri));
	cudaMemcpy(dev_tris, scene->mesh_tris_sorted.data(), scene->num_tris * sizeof(Tri), cudaMemcpyHostToDevice);*/

	cudaMalloc(&dev_tris, scene->triangles.size() * sizeof(Tri));
	cudaMemcpy(dev_tris, scene->triangles.data(), scene->triangles.size() * sizeof(Tri), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_lbvh, scene->lbvh.size() * sizeof(LBVHNode));
	cudaMemcpy(dev_lbvh, scene->lbvh.data(), scene->lbvh.size() * sizeof(LBVHNode), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));


	// FOR LIGHT SAMPLED MIS RAY
	cudaMalloc(&dev_direct_light_rays, pixelcount * sizeof(MISLightRay));

	cudaMalloc(&dev_direct_light_isects, pixelcount * sizeof(MISLightIntersection));
	cudaMemset(dev_direct_light_isects, 0, pixelcount * sizeof(MISLightIntersection));

	// FOR BSDF SAMPLED MIS RAY
	cudaMalloc(&dev_bsdf_light_rays, pixelcount * sizeof(MISLightRay));

	cudaMalloc(&dev_bsdf_light_isects, pixelcount * sizeof(MISLightIntersection));
	cudaMemset(dev_bsdf_light_isects, 0, pixelcount * sizeof(MISLightIntersection));

	// TODO: initialize any extra device memeory you need

}

void resetImage() {
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_tris);
	cudaFree(dev_lbvh);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_lights);
	cudaFree(dev_direct_light_rays);
	cudaFree(dev_direct_light_isects);
	cudaFree(dev_bsdf_light_rays);
	cudaFree(dev_bsdf_light_isects);
}

__global__ void generateRayFromThinLensCamera(Camera cam, int iter, int traceDepth, float jitterX, float jitterY, glm::vec3 thinLensCamOrigin, glm::vec3 newRef,
	PathSegment* pathSegments)
{
	__shared__ PathSegment mat[BLOCK_SIZE_2D][BLOCK_SIZE_2D];

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	if (x < cam.resolution.x && y < cam.resolution.y) {
		mat[threadIdx.x][threadIdx.y] = pathSegments[index];
		PathSegment& segment = mat[threadIdx.x][threadIdx.y];

		segment.ray.origin = thinLensCamOrigin;
		segment.rng_engine = makeSeededRandomEngine(iter, index, traceDepth);
		segment.rayThroughput = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.accumulatedIrradiance = glm::vec3(0.0f, 0.0f, 0.0f);
		segment.prev_hit_was_specular = false;

		float jittered_x = ((float)x) + jitterX;
		float jittered_y = ((float)y) + jitterY;

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(
			glm::normalize(newRef - thinLensCamOrigin) - cam.right * cam.pixelLength.x * (jittered_x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * (jittered_y - (float)cam.resolution.y * 0.5f)
		);

		segment.ray.direction_inv = 1.0f / segment.ray.direction;

		segment.remainingBounces = traceDepth;

		pathSegments[index] = mat[threadIdx.x][threadIdx.y];
	}
}

__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, float jitterX, float jitterY,
	PathSegment* pathSegments)
{
	__shared__ PathSegment mat[BLOCK_SIZE_2D][BLOCK_SIZE_2D];

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	if (x < cam.resolution.x && y < cam.resolution.y) {
		mat[threadIdx.x][threadIdx.y] = pathSegments[index];
		PathSegment& segment = mat[threadIdx.x][threadIdx.y];

		segment.ray.origin = cam.position;
		segment.rng_engine = makeSeededRandomEngine(iter, index, traceDepth);
		segment.rayThroughput = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.accumulatedIrradiance = glm::vec3(0.0f, 0.0f, 0.0f);
		segment.prev_hit_was_specular = false;

		float jittered_x = ((float)x) + jitterX;
		float jittered_y = ((float)y) + jitterY;

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(
			cam.view - cam.right * cam.pixelLength.x * (jittered_x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * (jittered_y - (float)cam.resolution.y * 0.5f)
		);

		segment.ray.direction_inv = 1.0f / segment.ray.direction;

		segment.remainingBounces = traceDepth;

		pathSegments[index] = mat[threadIdx.x][threadIdx.y];
	}
}

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, Tri* tris
	, int tris_size
	, ShadeableIntersection* intersections
	, LBVHNode* lbvh
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		if (pathSegments[path_index].remainingBounces == 0) {
			return;
		}
		Ray r = pathSegments[path_index].ray;

		ShadeableIntersection isect;
		isect.t = MAX_INTERSECT_DIST;

		float t;
		glm::vec3 tmp_normal;
		int obj_ID = -1;

		for (int i = 0; i < geoms_size; ++i)
		{
			Geom& geom = geoms[i];

			if (geom.type == SPHERE) {
#ifdef ENABLE_SPHERES
				t = sphereIntersectionTest(geom, r, tmp_normal);
#endif                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
			}
			else if (geom.type == SQUAREPLANE) {
#ifdef ENABLE_SQUAREPLANES
				t = squareplaneIntersectionTest(geom, r, tmp_normal);
#endif	
			}
			else if (geom.type == CUBE) {
#ifdef ENABLE_RECTS
				t = boxIntersectionTest(geom, r, tmp_normal);
#endif
			}
			else if (geom.type == MESH) {
#ifdef ENABLE_TRIS
				t = lbvhIntersectionTest(pathSegments[path_index], lbvh, tris, r, geom.triangleCount, tmp_normal, true);
#endif
			}

			if (depth == 0 && glm::dot(tmp_normal, r.direction) > 0.0) { 
				continue; 
			}
			else if (isect.t > t) {
				isect.t = t;
				isect.materialId = geom.materialid;
				isect.surfaceNormal = tmp_normal;
			}
			
		}

		if (isect.t >= MAX_INTERSECT_DIST) {
			// hits nothing
			pathSegments[path_index].remainingBounces = 0;
		}
		else {
			intersections[path_index] = isect;
		}
	}
}

__global__ void genMISRaysKernel(
	int iter
	, int num_paths
	, int max_depth
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, MISLightRay* direct_light_rays
	, MISLightRay* bsdf_light_rays
	, Light* lights
	, int num_lights
	, Geom* geoms
	, MISLightIntersection* direct_light_isects
	, MISLightIntersection* bsdf_light_isects
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces == 0) {
			return;
		}

		ShadeableIntersection intersection = shadeableIntersections[idx];
		Material material = materials[intersection.materialId];
		
		if (material.emittance > 0.0f) {
			if (pathSegments[idx].remainingBounces == max_depth || pathSegments[idx].prev_hit_was_specular) {
				// only color lights on first hit
				pathSegments[idx].accumulatedIrradiance += (material.R * material.emittance) * pathSegments[idx].rayThroughput;
			}
			pathSegments[idx].remainingBounces = 0;
			return;
		}

		pathSegments[idx].prev_hit_was_specular = material.type == SPEC_BRDF || material.type == SPEC_BTDF || material.type == SPEC_GLASS;

		if (pathSegments[idx].prev_hit_was_specular) {
			return;
		}

		glm::vec3 intersect_point = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;

		thrust::default_random_engine& rng = pathSegments[idx].rng_engine;
		thrust::uniform_real_distribution<float> u01(0, 1);

		// choose light to directly sample
		direct_light_rays[idx].light_ID = bsdf_light_rays[idx].light_ID = lights[glm::min((int)(glm::floor(u01(rng) * (float)num_lights)), num_lights - 1)].geom_ID;

		Geom& light = geoms[direct_light_rays[idx].light_ID];

		Material& light_material = materials[light.materialid];

		////////////////////////////////////////////////////
		// LIGHT SAMPLED
		////////////////////////////////////////////////////

		// generate light sampled wi
		glm::vec3 wi = glm::vec3(0.0f);
		float absDot = 0.0f;
		glm::vec3 f = glm::vec3(0.0f);
		float pdf_L = 0.0f;
		float pdf_B = 0.0f;

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
			pdf_B = absDot * 0.31831f;
			f = material.R * 0.31831f; // INV_PI
			 
		}
		direct_light_rays[idx].f = f;
		direct_light_rays[idx].pdf = pdf_B;

		// LTE = f * Li * absDot / pdf
		if (pdf_L <= 0.0001f) {
			direct_light_isects[idx].LTE = glm::vec3(0.0f, 0.0f, 0.0f);
		}
		else {
			direct_light_isects[idx].LTE = light_material.emittance * light_material.R * f * absDot / pdf_L;

		}

		// MIS Power Heuristic
		if (pdf_L <= 0.0001f && pdf_B <= 0.0001f) {
			direct_light_isects[idx].w = 0.0f;
		}
		else {
			direct_light_isects[idx].w = (pdf_L * pdf_L) / ((pdf_L * pdf_L) + (pdf_B * pdf_B));
		}


		////////////////////////////////////////////////////
		// BSDF SAMPLED
		////////////////////////////////////////////////////

		if (material.type == SPEC_BRDF) {
			// spec refl
			wi = glm::reflect(pathSegments[idx].ray.direction, intersection.surfaceNormal);
			absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
			pdf_B = 1.0f;
			if (absDot == 0.0f) {
				f = material.R;
			}
			else {
				f = material.R / absDot;
			}
		}
		else if (material.type == SPEC_BTDF) {
			// spec refr
			float eta = material.ior;
			if (glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction) < 0.0f) {
				// outside
				eta = 1.0f / eta;
				wi = glm::refract(pathSegments[idx].ray.direction, intersection.surfaceNormal, eta);
			}
			else {
				// inside
				wi = glm::refract(pathSegments[idx].ray.direction, -intersection.surfaceNormal, eta);
			}
			absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
			pdf_B = 1.0f;
			if (glm::length(wi) <= 0.0001f) {
				// total internal reflection
				f = glm::vec3(0.0f);
			}
			if (absDot == 0.0f) {
				f = material.T;
			}
			else {
				f = material.T / absDot;
			}
		}
		else if (material.type == SPEC_GLASS) {
			// spec glass
			float eta = material.ior;
			if (u01(rng) < 0.5f) {
				// spec refl
				wi = glm::reflect(pathSegments[idx].ray.direction, intersection.surfaceNormal);
				absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
				pdf_B = 1.0f;
				if (absDot == 0.0f) {
					f = material.R;
				}
				else {
					f = material.R / absDot;
				}
				f *= fresnelDielectric(glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction), material.ior);
			}
			else {
				// spec refr
				if (glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction) < 0.0f) {
					// outside
					eta = 1.0f / eta;
					wi = glm::refract(pathSegments[idx].ray.direction, intersection.surfaceNormal, eta);
				}
				else {
					// inside
					wi = glm::refract(pathSegments[idx].ray.direction, -intersection.surfaceNormal, eta);
				}
				absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
				pdf_B = 1.0f;
				if (glm::length(wi) <= 0.0001f) {
					// total internal reflection
					f = glm::vec3(0.0f);
				}
				else if (absDot == 0.0f) {
					f = material.T;
				}
				else {
					f = material.T / absDot;
				}
				f *= glm::vec3(1.0f) - fresnelDielectric(glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction), material.ior);
			}
			f *= 2.0f;
		}
		else {
			// diffuse
			wi = glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng, u01));
			if (material.type == DIFFUSE_BTDF) {
				wi = -wi;
			}
			absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
			pdf_B = absDot * 0.31831f;
			f = material.R * 0.31831f; // INV_PI
		}


		// Change ray direction
		bsdf_light_rays[idx].ray.origin = intersect_point + (wi * 0.001f);
		bsdf_light_rays[idx].ray.direction = wi;
		bsdf_light_rays[idx].ray.direction_inv = 1.0f / wi;
		bsdf_light_rays[idx].f = f;


		// LTE = f * Li * absDot / pdf
		absDot = glm::abs(glm::dot(intersection.surfaceNormal, bsdf_light_rays[idx].ray.direction));
		bsdf_light_rays[idx].pdf = pdf_B;

		if (pdf_B <= 0.0001f) {
			bsdf_light_isects[idx].LTE = glm::vec3(0.0f, 0.0f, 0.0f);
		}
		else {
			bsdf_light_isects[idx].LTE = light_material.emittance * light_material.R * bsdf_light_rays[idx].f * absDot / pdf_B;
		}
		
	}
}

__global__ void computeDirectLightIsects(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, MISLightRay* direct_light_rays
	, Geom* geoms
	, int geoms_size
	, Tri* tris
	, int tris_size
	, MISLightIntersection* direct_light_intersections
	, LBVHNode* lbvh
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{

		if (pathSegments[path_index].remainingBounces == 0) {
			return;
		}
		else if (pathSegments[path_index].prev_hit_was_specular) {
			return;
		}

		MISLightRay r = direct_light_rays[path_index];

		float t_min = MAX_INTERSECT_DIST;
		int obj_ID = -1;
		float t;
		glm::vec3 tmp_normal;

		for (int i = 0; i < geoms_size; ++i)
		{
			Geom& geom = geoms[i];

			if (geom.type == SPHERE) {
#ifdef ENABLE_SPHERES
				t = sphereIntersectionTest(geom, r.ray, tmp_normal);
#endif
			}
			else if (geom.type == SQUAREPLANE) {
#ifdef ENABLE_SQUAREPLANES
				t = squareplaneIntersectionTest(geom, r.ray, tmp_normal);
#endif
			}
			else if (geom.type == CUBE) {
#ifdef ENABLE_RECTS
				t = boxIntersectionTest(geom, r.ray, tmp_normal);
#endif
			}
			else if (geom.type == MESH) {
#ifdef ENABLE_TRIS
				t = lbvhIntersectionTest(pathSegments[path_index], lbvh, tris, r.ray, geom.triangleCount, tmp_normal, false);
#endif
			}

			if (t_min > t)
			{
				t_min = t;
				obj_ID = i;
			}
		}

		if (obj_ID != r.light_ID) {
			direct_light_intersections[path_index].LTE = glm::vec3(0.0f, 0.0f, 0.0f);
			direct_light_intersections[path_index].w = 0.0f;
		}

		// LTE = f * Li * absDot / pdf
		// Already have f, Li, absDot, and pdf from when we generated ray
		// MIS Power Heuristic already calulated in raygen
	}
}

__global__ void computeBSDFLightIsects(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, MISLightRay* bsdf_light_rays
	, Geom* geoms
	, int geoms_size
	, Tri* tris
	, int tris_size
	, MISLightIntersection* bsdf_light_intersections
	, LBVHNode* lbvh
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{

		if (pathSegments[path_index].remainingBounces == 0) {
			return;
		}
		else if (pathSegments[path_index].prev_hit_was_specular) {
			return;
		}

		MISLightRay r = bsdf_light_rays[path_index];

		float t_min = MAX_INTERSECT_DIST;
		int obj_ID = -1;
		float pdf_L_B = 0.0f;
		float t;
		glm::vec3 hit_normal;
		glm::vec3 tmp_normal;

		for (int i = 0; i < geoms_size; ++i)
		{
			Geom& geom = geoms[i];

			if (geom.type == SPHERE) {
#ifdef ENABLE_SPHERES
				t = sphereIntersectionTest(geom, r.ray, tmp_normal);
#endif
			}
			else if (geom.type == SQUAREPLANE) {
#ifdef ENABLE_SQUAREPLANES
				t = squareplaneIntersectionTest(geom, r.ray, tmp_normal);
#endif
			}
			else if (geom.type == CUBE) {
#ifdef ENABLE_RECTS
				t = boxIntersectionTest(geom, r.ray, tmp_normal);
#endif
			}
			else if (geom.type == MESH) {
#ifdef ENABLE_TRIS
				t = lbvhIntersectionTest(pathSegments[path_index], lbvh, tris, r.ray, geom.triangleCount, tmp_normal, false);
#endif
			}

			if (t_min > t)
			{
				hit_normal = tmp_normal;
				t_min = t;
				obj_ID = i;
			}
		}

		float absDot = glm::dot(hit_normal, r.ray.direction);

		if (obj_ID == r.light_ID && absDot < 0.0f) {

			absDot = glm::abs(absDot);
			pdf_L_B = (t_min * t_min) / (absDot * geoms[obj_ID].scale.x * geoms[obj_ID].scale.y);

			// LTE = f * Li * absDot / pdf
			// Already have f, Li, and pdf from when we generated ray
			bsdf_light_intersections[path_index].LTE *= absDot;

			// MIS Power Heuristic
			if (pdf_L_B == 0.0f && r.pdf == 0.0f) {
				bsdf_light_intersections[path_index].w = 0.0f;
			}
			else {
				bsdf_light_intersections[path_index].w = (r.pdf * r.pdf) / ((r.pdf * r.pdf) + (pdf_L_B * pdf_L_B));
			}
		}
		else {
			bsdf_light_intersections[path_index].LTE = glm::vec3(0.0f, 0.0f, 0.0f);
			bsdf_light_intersections[path_index].w = 0.0f;
		}
	}
}

__global__ void shadeMaterialUberKernel(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, MISLightIntersection* direct_light_isects
	, MISLightIntersection* bsdf_light_isects
	, int num_lights
	, PathSegment* pathSegments
	, Material* materials
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces == 0) {
			return;
		}
		ShadeableIntersection intersection = shadeableIntersections[idx];
		MISLightIntersection direct_light_intersection = direct_light_isects[idx];
		MISLightIntersection bsdf_light_intersection = bsdf_light_isects[idx];

		thrust::default_random_engine& rng = pathSegments[idx].rng_engine;
		thrust::uniform_real_distribution<float> u01(0.0, 1.0);

		Material m = materials[intersection.materialId];

		glm::vec3 intersect_point = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction;

		// Combine direct light and bsdf light samples with Power Heuristic
		if (!pathSegments[idx].prev_hit_was_specular) {
			pathSegments[idx].accumulatedIrradiance += pathSegments[idx].rayThroughput * (float)num_lights *
				(direct_light_intersection.w * direct_light_intersection.LTE +
					bsdf_light_intersection.w * bsdf_light_intersection.LTE);
		}

		glm::vec3 wi = glm::vec3(0.0f);
		float pdf = 0.0f;
		float absDot = 0.0f;
		glm::vec3 f = Sample_f(m, pathSegments[idx], intersection, &wi, &pdf, absDot, rng, u01);
		pathSegments[idx].rayThroughput *= f * absDot / pdf;

		// Change ray direction
		pathSegments[idx].ray.direction = wi;
		pathSegments[idx].ray.direction_inv = 1.0f / wi;
		pathSegments[idx].ray.origin = intersect_point + (wi * 0.001f);
		pathSegments[idx].remainingBounces--;
	}
}

__global__ void russianRouletteKernel(int iter, int num_paths, PathSegment* pathSegments)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces == 0) {
			return;
		}
		thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
		thrust::default_random_engine& rng = pathSegments[idx].rng_engine;
		float random_num = u01(rng);
		float max_channel = glm::max(glm::max(pathSegments[idx].rayThroughput.r, pathSegments[idx].rayThroughput.g), pathSegments[idx].rayThroughput.b);
		if (max_channel < random_num) {
			pathSegments[idx].remainingBounces = 0;
		}
		else {
			pathSegments[idx].rayThroughput /= max_channel;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[index] += iterationPath.accumulatedIrradiance;
	}
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		pix /= iter;

		// reinhard (HDR)
		pix /= (pix + glm::vec3(1.0f));

		// gamma correction
		pix = glm::pow(pix, glm::vec3(0.454545f));

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}


struct is_done
{
	__host__ __device__
		bool operator()(const PathSegment &path)
	{
		return path.remainingBounces != 0;
	}
};

struct material_sort
{
	__host__ __device__
		bool operator()(const ShadeableIntersection& isect_0, const ShadeableIntersection& isect_1)
	{
		return isect_0.materialId < isect_1.materialId;
	}
};

void pathtrace(uchar4* pbo, int frame, int iter) {

	
	//std::cout << "============================== " << iter << " ==============================" << std::endl;

	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D,
		(cam.resolution.y + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D);


	// 1D block for path tracing
	const int blockSize1d = BLOCK_SIZE_1D;

	int depth = 0;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;

	dim3 numblocksPathSegmentTracing = (pixelcount + blockSize1d - 1) / blockSize1d;


	// gen ray
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, iter, iter);
	thrust::uniform_real_distribution<float> upixel(0.0, 1.0f);

	float jitterX = upixel(rng);
	float jitterY = upixel(rng);

	if (cam.lens_radius > 0.0f) {
		// thin lens camera model based on my implementation from CIS 561
		// also based on https://www.semanticscholar.org/paper/A-Low-Distortion-Map-Between-Disk-and-Square-Shirley-Chiu/43226a3916a85025acbb3a58c17f6dc0756b35ac?p2df
		glm::mat3 M = glm::mat3(cam.right, cam.up, cam.view);

		float focalT = (cam.focal_distance / glm::length(cam.lookAt - cam.position));
		glm::vec3 newRef = cam.position + focalT * (cam.lookAt - cam.position);
		glm::vec2 thinLensSample = glm::vec2(upixel(rng), upixel(rng));

		// turn square shaped random sample domain into disc shaped
		glm::vec3 warped = glm::vec3(0.0f);
		glm::vec2 sampleRemap = 2.0f * thinLensSample - glm::vec2(1.0f);
		float r, theta = 0.0f;
		if (glm::abs(sampleRemap.x) > glm::abs(sampleRemap.y)) {
			r = sampleRemap.x;
			theta = (PI / 4.0f) * (sampleRemap.y / sampleRemap.x);
		}
		else {
			r = sampleRemap.y;
			theta = (PI / 2.0f) - (PI / 4.0f) * (sampleRemap.x / sampleRemap.y);
		}
		warped = r * glm::vec3(glm::cos(theta), glm::sin(theta), 0.0f);

		glm::vec3 lensPoint = cam.lens_radius * warped;

		glm::vec3 thinLensCamOrigin = cam.position + M * lensPoint;

		generateRayFromThinLensCamera << <blocksPerGrid2d, blockSize2d >> > (cam,
			iter, traceDepth, jitterX, jitterY, thinLensCamOrigin, newRef, dev_paths);
	}
	else {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam,
			iter, traceDepth, jitterX, jitterY, dev_paths);
	}

	while (!iterationComplete) {
		//std::cout << "depth: " << depth << std::endl;

		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, pixelcount
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_tris
			, hst_scene->num_tris
			, dev_intersections
			, dev_lbvh
			);

		depth++;

		genMISRaysKernel << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			pixelcount,
			traceDepth,
			dev_intersections,
			dev_paths,
			dev_materials,
			dev_direct_light_rays,
			dev_bsdf_light_rays,
			dev_lights,
			hst_scene->lights.size(),
			dev_geoms,
			dev_direct_light_isects,
			dev_bsdf_light_isects
			);

		computeDirectLightIsects << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, pixelcount
			, dev_paths
			, dev_direct_light_rays
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_tris
			, hst_scene->num_tris
			, dev_direct_light_isects
			, dev_lbvh
			);

		computeBSDFLightIsects << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, pixelcount
			, dev_paths
			, dev_bsdf_light_rays
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_tris
			, hst_scene->num_tris
			, dev_bsdf_light_isects
			, dev_lbvh
			);

		shadeMaterialUberKernel << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			pixelcount,
			dev_intersections,
			dev_direct_light_isects,
			dev_bsdf_light_isects,
			hst_scene->lights.size(),
			dev_paths,
			dev_materials
			);

		// RUSSIAN ROULETTE
		if (depth >= 5) {

			russianRouletteKernel << <numblocksPathSegmentTracing, blockSize1d >> > (
				iter,
				pixelcount,
				dev_paths
				);

		}

		if (depth == traceDepth) { iterationComplete = true; }
	}


	// Assemble this iteration and apply it to the image
	finalGather << <numblocksPathSegmentTracing, blockSize1d >> > (pixelcount, dev_image, dev_paths);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}