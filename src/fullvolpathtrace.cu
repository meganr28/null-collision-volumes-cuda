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
#include "fullVolPathtrace.h"
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

#define BOUNCE_PADDING 128



#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn_FullVol(msg, FILENAME, __LINE__)
void checkCUDAErrorFn_FullVol(const char* msg, const char* file, int line) {
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
thrust::default_random_engine makeSeededRandomEngine_FullVol(int iter, int index, int depth) {
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
static BVHNode_GPU* dev_bvh_nodes = NULL;
static Medium* dev_media = NULL;
static nanovdb::NanoGrid<float>* dev_media_density = NULL;
//cudaStream_t media_stream;

static MISLightRay* dev_direct_light_rays = NULL;
static MISLightIntersection* dev_direct_light_isects = NULL;

static MISLightRay* dev_bsdf_light_rays = NULL;
static MISLightIntersection* dev_bsdf_light_isects = NULL;

static glm::vec3* dev_sample_colors = NULL;

int pixelcount_fullvol;

// TODO: remove these when done testing
__global__ void grid_test_kernel_FullVol(const nanovdb::NanoGrid<float>* deviceGrid)
{
	if (threadIdx.x > 6)
		return;
	int i = 97 + threadIdx.x;
	auto gpuAcc = deviceGrid->getAccessor();
	printf("(%3i,0,0) NanoVDB gpu: % -4.2f\n", i, gpuAcc.getValue(nanovdb::Coord(i, i, i)));
}

void fullVolResetImage() {
	cudaMemset(dev_image, 0, pixelcount_fullvol * sizeof(glm::vec3));
}

void fullVolPathtraceInit(Scene* scene) {

	hst_scene = scene;
	
	const Camera& cam = hst_scene->state.camera;
	pixelcount_fullvol = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount_fullvol * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount_fullvol * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount_fullvol * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	/*cudaMalloc(&dev_tris, scene->num_tris * sizeof(Tri));
	cudaMemcpy(dev_tris, scene->mesh_tris_sorted.data(), scene->num_tris * sizeof(Tri), cudaMemcpyHostToDevice);*/

	cudaMalloc(&dev_tris, scene->triangles.size() * sizeof(Tri));
	cudaMemcpy(dev_tris, scene->triangles.data(), scene->triangles.size() * sizeof(Tri), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_lbvh, scene->lbvh.size() * sizeof(LBVHNode));
	cudaMemcpy(dev_lbvh, scene->lbvh.data(), scene->lbvh.size() * sizeof(LBVHNode), cudaMemcpyHostToDevice);
	
	cudaMalloc(&dev_bvh_nodes, scene->bvh_nodes_gpu.size() * sizeof(BVHNode_GPU));
	cudaMemcpy(dev_bvh_nodes, scene->bvh_nodes_gpu.data(), scene->bvh_nodes_gpu.size() * sizeof(BVHNode_GPU), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount_fullvol * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount_fullvol * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_media, scene->media.size() * sizeof(Medium));
	cudaMemcpy(dev_media, scene->media.data(), scene->media.size() * sizeof(Medium), cudaMemcpyHostToDevice);

	// Copy NanoVDB grid to the GPU
	scene->gridHandle.deviceUpload();
	dev_media_density = scene->gridHandle.deviceGrid<float>();
	//grid_test_kernel <<< 1, 64 >>> (dev_media_density);
	
	// Copy NanoVDB grid to the GPU asynchronously (for later)
	//cudaStreamCreate(&media_stream);
	//scene->gridHandle.deviceUpload(media_stream, false);
	
	// FOR LIGHT SAMPLED MIS RAY
	cudaMalloc(&dev_direct_light_rays, pixelcount_fullvol * sizeof(MISLightRay));

	cudaMalloc(&dev_direct_light_isects, pixelcount_fullvol * sizeof(MISLightIntersection));
	cudaMemset(dev_direct_light_isects, 0, pixelcount_fullvol * sizeof(MISLightIntersection));

	// FOR BSDF SAMPLED MIS RAY
	cudaMalloc(&dev_bsdf_light_rays, pixelcount_fullvol * sizeof(MISLightRay));

	cudaMalloc(&dev_bsdf_light_isects, pixelcount_fullvol * sizeof(MISLightIntersection));
	cudaMemset(dev_bsdf_light_isects, 0, pixelcount_fullvol * sizeof(MISLightIntersection));

	// TODO: initialize any extra device memeory you need

}

void fullVolPathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_tris);
	cudaFree(dev_lbvh);
	cudaFree(dev_bvh_nodes);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_lights);
	cudaFree(dev_direct_light_rays);
	cudaFree(dev_direct_light_isects);
	cudaFree(dev_bsdf_light_rays);
	cudaFree(dev_bsdf_light_isects);
}

__global__ void generateRayFromThinLensCamera_FullVol(Camera cam, int iter, int traceDepth, float jitterX, float jitterY, glm::vec3 thinLensCamOrigin, glm::vec3 newRef,
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
		segment.rng_engine = makeSeededRandomEngine_FullVol(iter, index, traceDepth);
		segment.rayThroughput = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.r_u = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.r_l = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.accumulatedIrradiance = glm::vec3(0.0f, 0.0f, 0.0f);
		segment.prev_hit_was_specular = false;
		segment.prev_hit_null_material = false;

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

__global__ void generateRayFromCamera_FullVol(Camera cam, int iter, int traceDepth, float jitterX, float jitterY,
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
		segment.rng_engine = makeSeededRandomEngine_FullVol(iter, index, traceDepth);
		segment.rayThroughput = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.r_u = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.r_l = glm::vec3(1.0f, 1.0f, 1.0f);
		segment.accumulatedIrradiance = glm::vec3(0.0f, 0.0f, 0.0f);
		segment.prev_hit_was_specular = false;
		segment.prev_hit_null_material = false;
		segment.prev_event_was_real = true;
		segment.medium = cam.medium;

		float jittered_x = ((float)x) + jitterX;
		float jittered_y = ((float)y) + jitterY;

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(
			cam.view - cam.right * cam.pixelLength.x * (jittered_x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * (jittered_y - (float)cam.resolution.y * 0.5f)
		);

		segment.ray.direction_inv = 1.0f / segment.ray.direction;
		segment.lastRealRay = segment.ray;

		segment.remainingBounces = traceDepth;
		segment.realPathLength = 0;

		pathSegments[index] = mat[threadIdx.x][threadIdx.y];
	}
}

__global__ void computeIntersections_FullVol(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, Tri* tris
	, int tris_size
	, Medium* media
	, int media_size
	, ShadeableIntersection* intersections
	, LBVHNode* lbvh
	, BVHNode_GPU* bvh_nodes
)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		if (pathSegments[path_index].remainingBounces <= 0) {
			return;
		}
		Ray r = pathSegments[path_index].ray;

		ShadeableIntersection isect;
		isect.objID = -1;
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
				isect.objID = i;
				isect.materialId = geom.materialid;
				isect.surfaceNormal = tmp_normal;

				// Check if surface is medium transition
				if (IsMediumTransition(geom.mediumInterface)) {
					isect.mediumInterface = geom.mediumInterface;
				}
				else {
					isect.mediumInterface.inside = pathSegments[path_index].medium;
					isect.mediumInterface.outside = pathSegments[path_index].medium;
				}
			}
			
		}

		for (int j = 0; j < media_size; j++) {
			if (media[j].type == HOMOGENEOUS) continue;

			const Medium& medium = media[j];
			float tMin, tMax;
			bool intersectAABB = aabbIntersectionTest(pathSegments[path_index], medium.aabb_min, medium.aabb_max, pathSegments[path_index].ray, tMin, tMax, t, false);

			if (intersectAABB && isect.t > t) {
				isect.t = t;
				isect.materialId = -1;
				isect.surfaceNormal = glm::vec3(0.0f);

				// TODO: change this to handle more advanced cases
				isect.mediumInterface.inside = j;
				isect.mediumInterface.outside = -1;

				isect.tMin = tMin;
				isect.tMax = tMax;
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

__global__ void sampleParticipatingMedium_FullVol(
	int num_paths,
	int max_depth,
	int depth,
	PathSegment* pathSegments,
	Material* materials,
	ShadeableIntersection* intersections,
	Geom* geoms,
	int geoms_size,
	Tri* tris,
	int tris_size,
	Medium* media,
	int media_size,
	MISLightRay* direct_light_rays,
	MISLightIntersection* direct_light_isects,
	Light* lights,
	int num_lights,
	LBVHNode* lbvh,
	BVHNode_GPU* bvh_nodes,
	const nanovdb::NanoGrid<float>* media_density,
	GuiParameters gui_params)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces <= 0) {
			return;
		}

		if (glm::isnan(pathSegments[idx].accumulatedIrradiance.x) || glm::isnan(pathSegments[idx].accumulatedIrradiance.y) || glm::isnan(pathSegments[idx].accumulatedIrradiance.z)) {
			pathSegments[idx].accumulatedIrradiance = glm::vec3(100000.0, 0.0, 0.0);
			pathSegments[idx].rayThroughput = glm::vec3(10.0, 0.0, 0.0);
			pathSegments[idx].remainingBounces = 0;
			return;
		}

		/*if (depth == max_depth - 3 && pathSegments[idx].remainingBounces > 0) {
			pathSegments[idx].accumulatedIrradiance += glm::vec3(0, 0, 1);
		}*/

		//// Ray from last real collision
		//if (pathSegments[idx].prev_event_was_real) {
		//	pathSegments[idx].lastRealRay = pathSegments[idx].ray;
		//}

		thrust::default_random_engine& rng = pathSegments[idx].rng_engine;
		thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

		// If we have a medium, sample participating medium
		int rayMediumIndex = pathSegments[idx].medium;
		MediumInteraction mi;
		mi.medium = -1;
		glm::vec3 T_maj;
		if (rayMediumIndex >= 0) {
			if (media[rayMediumIndex].type == HOMOGENEOUS) {
				pathSegments[idx].rayThroughput *= Sample_homogeneous(media[rayMediumIndex], pathSegments[idx], intersections[idx], &mi, rayMediumIndex, u01(rng));
			}
			else {
				//pathSegments[idx].rayThroughput *= Sample_heterogeneous(media[rayMediumIndex], pathSegments[idx], intersections[idx], &mi, media_density, rayMediumIndex, rng, u01);
				T_maj = Sample_channel(max_depth, media[rayMediumIndex], pathSegments[idx], intersections[idx], &mi, media_density, rayMediumIndex, gui_params, rng, u01);
				//if (glm::length(T_maj) < 0.f) pathSegments[idx].accumulatedIrradiance += glm::vec3(1.0, 0.0, 0.0);
			}
		}
		if (glm::length(pathSegments[idx].rayThroughput) <= 0.0f) {
			pathSegments[idx].remainingBounces = 0;
		}
		intersections[idx].mi = mi;

		// Handle medium interaction
		bool scattered = false;
		if (mi.medium >= 0) {
			//pathSegments[idx].rayThroughput *= handleMediumInteraction(max_depth, media[rayMediumIndex], pathSegments[idx], intersections[idx], mi, media_density, rayMediumIndex, rng, u01);
			scattered = handleMediumInteraction(idx, max_depth, T_maj, pathSegments, materials, intersections[idx], mi, geoms, geoms_size, tris, tris_size,
				media, media_size, media_density, direct_light_rays, direct_light_isects, lights, num_lights, lbvh, bvh_nodes, gui_params, rng, u01);
		}

		if (pathSegments[idx].remainingBounces <= 0) {
			return;
		}

		if (scattered) {
			return;
		}


		if (rayMediumIndex >= 0) {
			pathSegments[idx].rayThroughput *= T_maj / T_maj[0];
			pathSegments[idx].r_l *= T_maj / T_maj[0];
			pathSegments[idx].r_u *= T_maj / T_maj[0];
		}

		// Handle surface interaction
		ShadeableIntersection intersection = intersections[idx];

		// hit an invisible bounding surface
		if (intersections[idx].mi.medium == -1) {
			if (intersection.materialId < 0) {
				// Change ray direction
				pathSegments[idx].ray.origin = pathSegments[idx].ray.origin + ((intersection.t + 0.001f) * pathSegments[idx].ray.direction);
				//pathSegments[idx].medium = glm::dot(pathSegments[idx].ray.direction, intersection.surfaceNormal) > 0 ? intersection.mediumInterface.outside : intersection.mediumInterface.inside;
				
				// TODO make work for both volume trypes
				//if (glm::dot(pathSegments[idx].ray.direction, intersection.surfaceNormal) > -0.01f && glm::dot(pathSegments[idx].ray.direction, intersection.surfaceNormal) < 0.01f) pathSegments[idx].accumulatedIrradiance += glm::vec3(1.0, 0.0, 0.0);
				pathSegments[idx].medium = insideMedium(pathSegments[idx], intersection.tMin, intersection.tMax, 0) ? intersection.mediumInterface.inside : intersection.mediumInterface.outside;
				
				//pathSegments[idx].remainingBounces--;
				pathSegments[idx].prev_hit_null_material = true;
				return;
			}
		}

		Material material = materials[intersection.materialId];


		// Hit a light
		if (intersections[idx].mi.medium == -1) {
			
			if (material.emittance > 0.0f) {
				if (pathSegments[idx].remainingBounces == max_depth || pathSegments[idx].prev_hit_was_specular) {
					// only color lights on first hit
					pathSegments[idx].accumulatedIrradiance += (material.R * material.emittance) * pathSegments[idx].rayThroughput / pathSegments[idx].r_u;
				}
				else {
					//pathSegments[idx].accumulatedIrradiance += glm::vec3(1, 0, 0);
					if (glm::dot(intersection.surfaceNormal, glm::normalize(pathSegments[idx].ray.direction)) > 0.0001f) {

					}
					else {
						int light_ID = -1;
						for (int light_iter = 0; light_iter < num_lights; light_iter++) {
							if (lights[light_iter].geom_ID == intersection.objID) {
								light_ID = light_iter;
								break;
							}
						}
						float dist = glm::length(pathSegments[idx].ray.origin - (pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction));
						float pdf_L = (intersection.t * intersection.t) / (glm::abs(glm::dot(intersection.surfaceNormal, glm::normalize(pathSegments[idx].ray.direction))) * geoms[intersection.objID].scale.x * geoms[intersection.objID].scale.y);
						pdf_L *= (1.0f / (float)num_lights);
						pathSegments[idx].r_l *= pdf_L;
						pathSegments[idx].accumulatedIrradiance += (material.R * material.emittance) * pathSegments[idx].rayThroughput / (pathSegments[idx].r_u + pathSegments[idx].r_l);
					}
				}
				pathSegments[idx].remainingBounces = 0;
				return;
			}

			pathSegments[idx].prev_hit_was_specular = material.type == SPEC_BRDF || material.type == SPEC_BTDF || material.type == SPEC_GLASS;


		}


		// hit a normal surface
		if (intersections[idx].mi.medium == -1) {


			
			if (!pathSegments[idx].prev_hit_was_specular) {

				glm::vec3 Ld = directLightSample(idx, false, pathSegments, materials, intersection, geoms, geoms_size, tris, tris_size,
					media, media_size, media_density, direct_light_rays, direct_light_isects, lights, num_lights, lbvh, bvh_nodes, gui_params, rng, u01);

				pathSegments[idx].accumulatedIrradiance += pathSegments[idx].rayThroughput * Ld;

			}

			glm::vec3 wi = glm::vec3(0.0f);
			glm::vec3 f = glm::vec3(0.0f);
			float pdf = 0.0f;
			float absDot = 0.0f;

			// Physically based BSDF sampling influenced by PBRT
			// https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission
			// https://www.pbr-book.org/3ed-2018/Reflection_Models/Lambertian_Reflection

			if (material.type == SPEC_BRDF) {
				wi = glm::reflect(pathSegments[idx].ray.direction, intersection.surfaceNormal);
				absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
				pdf = 1.0f;
				if (absDot >= -0.0001f && absDot <= -0.0001f) {
					f = material.R;
				}
				else {
					f = material.R / absDot;
				}
			}
			else if (material.type == SPEC_BTDF) {
				// spec refl
				float eta = material.ior;
				if (glm::dot(intersection.surfaceNormal, pathSegments[idx].ray.direction) < 0.0001f) {
					// outside
					eta = 1.0f / eta;
					wi = glm::refract(pathSegments[idx].ray.direction, intersection.surfaceNormal, eta);
				}
				else {
					// inside
					wi = glm::refract(pathSegments[idx].ray.direction, -intersection.surfaceNormal, eta);
				}
				absDot = glm::abs(glm::dot(intersection.surfaceNormal, wi));
				pdf = 1.0f;
				if (glm::length(wi) <= 0.0001f) {
					// total internal reflection
					f = glm::vec3(0.0f);
				}
				else if (absDot >= -0.0001f && absDot <= -0.0001f) {
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
					pdf = 1.0f;
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
					pdf = 1.0f;
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
				pdf = absDot * 0.31831f;
				f = material.R * 0.31831f;
			}

			pathSegments[idx].rayThroughput *= f * absDot / pdf;

			
			// Change ray direction
			pathSegments[idx].r_l = pathSegments[idx].r_u / pdf;
			pathSegments[idx].ray.origin = (pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction) + (wi * 0.001f);
			pathSegments[idx].ray.direction = wi;
			pathSegments[idx].ray.direction_inv = 1.0f / wi;
			pathSegments[idx].medium = glm::dot(pathSegments[idx].ray.direction, intersection.surfaceNormal) > 0 ? intersection.mediumInterface.outside :
				intersection.mediumInterface.inside; // TODO: change for hetero
			pathSegments[idx].remainingBounces--;
		}
	}
}

__global__ void russianRouletteKernel_FullVol(int iter, int num_paths, PathSegment* pathSegments)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces <= 0) {
			return;
		}

		if (pathSegments[idx].remainingBounces > 4) {
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
}

// Add the current iteration's output to the overall image
__global__ void finalGather_FullVol(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[index] += iterationPath.accumulatedIrradiance;
	}
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO_FullVol(uchar4* pbo, glm::ivec2 resolution,
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
		return path.remainingBounces > 0;
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

void fullVolPathtrace(uchar4* pbo, int frame, int iter, GuiParameters& gui_params, int depth_padding, int refresh_rate, int refresh_bit) {

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

	dim3 numblocksPathSegmentTracing = (pixelcount_fullvol + blockSize1d - 1) / blockSize1d;

	// gen ray
	thrust::default_random_engine rng = makeSeededRandomEngine_FullVol(iter, iter, iter);
	thrust::uniform_real_distribution<float> upixel(0.0, 1.0f);

	float jitterX = upixel(rng);
	float jitterY = upixel(rng);

	generateRayFromCamera_FullVol << <blocksPerGrid2d, blockSize2d >> > (cam,
		iter, traceDepth, jitterX, jitterY, dev_paths);

	while (!iterationComplete) {
		//std::cout << "depth: " << depth << std::endl;
		// When intersecting with primitive, determine if there is a medium transition or not
		// Update isect struct's mediumInterface member variable with the appropriate mediumInterface
		computeIntersections_FullVol << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, pixelcount_fullvol
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_tris
			, hst_scene->num_tris
			, dev_media
			, hst_scene->media.size()
			, dev_intersections
			, dev_lbvh
			, dev_bvh_nodes
			);

		depth++;
		
		// Attenuating ray throughput with medium stuff (phase function)
		// Check if throughput is black, and break out of loop (set remainingBounces to 0)
		// If medium interaction is valid, then sample light and pick new direction by sampling phase function distribution
		// Else, handle surface interaction
		sampleParticipatingMedium_FullVol << <numblocksPathSegmentTracing, blockSize1d >> > (
			pixelcount_fullvol,
			traceDepth,
			depth,
			dev_paths,
			dev_materials,
			dev_intersections,
			dev_geoms,
			hst_scene->geoms.size(),
			dev_tris,
			hst_scene->num_tris,
			dev_media,
			hst_scene->media.size(),
			dev_direct_light_rays,
			dev_direct_light_isects,
			dev_lights,
			hst_scene->lights.size(),
			dev_lbvh,
			dev_bvh_nodes,
			dev_media_density,
			gui_params);
		
		// RUSSIAN ROULETTE
		if (depth > 4)
		{
			russianRouletteKernel_FullVol << <numblocksPathSegmentTracing, blockSize1d >> > (
				iter,
				pixelcount_fullvol,
				dev_paths
				);
		}

		if (depth == traceDepth + depth_padding) { iterationComplete = true; }
	}


	// Assemble this iteration and apply it to the image
	finalGather_FullVol << <numblocksPathSegmentTracing, blockSize1d >> > (pixelcount_fullvol, dev_image, dev_paths);


	if ((iter & refresh_rate) >> refresh_bit || iter < 2) {
		// 	// Send results to OpenGL buffer for rendering
		sendImageToPBO_FullVol << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

		// Retrieve image from GPU
		cudaMemcpy(hst_scene->state.image.data(), dev_image,
			pixelcount_fullvol * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	}

}