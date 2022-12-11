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
//#define STREAM_COMPACTION
//#define MEDIUM_SORT

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

PerformanceTimer& timer()
{
	static PerformanceTimer timer;
	return timer;
}

template<typename T>
void printElapsedTime(T time, std::string note = "")
{
	std::cout << "   elapsed time: " << time << "ms    " << note << std::endl;
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
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_lights);
	cudaFree(dev_direct_light_rays);
	cudaFree(dev_direct_light_isects);
	cudaFree(dev_bsdf_light_rays);
	cudaFree(dev_bsdf_light_isects);
}

__host__ __device__ int pickRGBWavelength(float rand)
{
	int num_samples = 3;
	return glm::min((int)(glm::floor(rand * (float)num_samples)), num_samples - 1);
}

/**
* Concentric Disk Sampling from PBRT Chapter 13.6.2
*/
__host__ __device__ glm::vec3 concentricSampleDisk_FullVol(glm::vec2& sample)
{
	// Map sample point (uniform random numbers) to range [-1, 1]
	glm::vec2 mappedSample = 2.f * sample - glm::vec2(1.f, 1.f);

	// Handle origin to avoid divide by zero
	if (mappedSample.x == 0.f && mappedSample.y == 0.f) {
		return glm::vec3(0.f);
	}

	// Apply concentric mapping to the adjusted sample point
	float r = 0.f;
	float theta = 0.f;
	// Find r and theta depending on x and y coords of mapped point
	if (std::abs(mappedSample.x) > std::abs(mappedSample.y)) {
		r = mappedSample.x;
		theta = (PI / 4.0f) * (mappedSample.y / mappedSample.x);
	}
	else {
		r = mappedSample.y;
		theta = (PI / 2.0f) - (PI / 4.0f) * (mappedSample.x / mappedSample.y);
	}

	return glm::vec3(r * glm::cos(theta), r * glm::sin(theta), 0.0f);
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

__global__ void generateRayFromCamera_FullVol(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	__shared__ PathSegment mat[BLOCK_SIZE_2D][BLOCK_SIZE_2D];

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * cam.resolution.x);

	thrust::uniform_real_distribution<float> upixel(0.0, 1.0f);

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
		segment.medium = cam.medium;
		//segment.rgbWavelength = pickRGBWavelength(upixel(segment.rng_engine));
		segment.rgbWavelength = 0;

		float jitterX = upixel(segment.rng_engine);
		float jitterY = upixel(segment.rng_engine);

		float jittered_x = ((float)x) + jitterX;
		float jittered_y = ((float)y) + jitterY;

		// Add antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(
			cam.view - cam.right * cam.pixelLength.x * (jittered_x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * (jittered_y - (float)cam.resolution.y * 0.5f)
		);

		// Add depth of field
		if (cam.lens_radius > 0.0f) {
			// Get sample on lens
			glm::vec2 thinLensSample = glm::vec2(upixel(segment.rng_engine), upixel(segment.rng_engine));
			glm::vec3 lensPoint = cam.lens_radius * concentricSampleDisk_FullVol(thinLensSample);

			// Get focal point
			float focalT = (cam.focal_distance / glm::length(cam.lookAt - cam.position));
			glm::vec3 newRef = segment.ray.origin + focalT * (cam.lookAt - cam.position);

			// Update ray
			segment.ray.origin += lensPoint;
			glm::vec3 newView = glm::normalize(newRef - segment.ray.origin);
			segment.ray.direction = glm::normalize(
				newView - cam.right * cam.pixelLength.x * (jittered_x - (float)cam.resolution.x * 0.5f)
				- cam.up * cam.pixelLength.y * (jittered_y - (float)cam.resolution.y * 0.5f)
			);
		}

		segment.ray.direction_inv = 1.0f / segment.ray.direction;
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;

		pathSegments[index] = mat[threadIdx.x][threadIdx.y];
	}
}

__global__ void computeIntersections_FullVol(
	int num_paths
	, int depth
	, PathSegment* pathSegments
	, Geom* geoms
	, Tri* tris
	, Medium* media
	, ShadeableIntersection* intersections
	, LBVHNode* lbvh
	, SceneInfo scene_info
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

		for (int i = 0; i < scene_info.geoms_size; ++i)
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

		if (scene_info.media_size > 0) {
			for (int j = 0; j < scene_info.media_size; j++) {
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
	int depth,
	PathSegment* pathSegments,
	Material* materials,
	ShadeableIntersection* intersections,
	Geom* geoms,
	Tri* tris,
	Medium* media,
	const nanovdb::NanoGrid<float>* media_density,
	MISLightRay* direct_light_rays,
	MISLightIntersection* direct_light_isects,
	Light* lights,
	LBVHNode* lbvh,
	GuiParameters gui_params,
	SceneInfo scene_info)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment& segment = pathSegments[idx];
		if (segment.remainingBounces <= 0) {
			return;
		}

		thrust::default_random_engine& rng = segment.rng_engine;
		thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

		// If we have a medium, sample participating medium
		int rayMediumIndex = segment.medium;
		MediumInteraction mi;
		mi.medium = -1;
		glm::vec3 T_maj;
		bool scattered = false;
		if (rayMediumIndex >= 0) {
			if (media[rayMediumIndex].type == HOMOGENEOUS) {
				segment.rayThroughput *= Sample_homogeneous(media[rayMediumIndex], segment, intersections[idx], &mi, rayMediumIndex, u01(rng));
			}
			else {
				T_maj = Sample_channel(
					idx,
					rayMediumIndex, 
					pathSegments[idx], 
					intersections[idx], 
					direct_light_rays[idx], 
					direct_light_isects[idx], 
					geoms, tris, lights, media, 
					materials, &mi, lbvh, media_density, 
					gui_params, scene_info, rng, u01, scattered);
			}
		}
		if (glm::length(segment.rayThroughput) <= 0.0f) {
			segment.remainingBounces = 0;
			return;
		}
		intersections[idx].mi = mi;

		if (segment.remainingBounces <= 0) {
			return;
		}

		if (scattered) {
			return;
		}

		if (rayMediumIndex >= 0) {
			segment.rayThroughput *= T_maj / T_maj[segment.rgbWavelength];
			segment.r_l *= T_maj / T_maj[segment.rgbWavelength];
			segment.r_u *= T_maj / T_maj[segment.rgbWavelength];
		}
	}
}

__global__ void handleSurfaceInteraction_FullVol(
	int num_paths,
	int depth,
	PathSegment* pathSegments,
	Material* materials,
	ShadeableIntersection* intersections,
	Geom* geoms,
	Tri* tris,
	Medium* media,
	const nanovdb::NanoGrid<float>* media_density,
	MISLightRay* direct_light_rays,
	MISLightIntersection* direct_light_isects,
	Light* lights,
	LBVHNode* lbvh,
	GuiParameters gui_params,
	SceneInfo scene_info)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces <= 0) {
			return;
		}

		if (intersections[idx].mi.medium >= 0) {
			return;
		}

		thrust::default_random_engine& rng = pathSegments[idx].rng_engine;
		thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

		// Handle surface interaction
		ShadeableIntersection intersection = intersections[idx];

		// hit an invisible bounding surface
		if (intersection.materialId < 0) {
			// Change ray direction
			pathSegments[idx].ray.origin = pathSegments[idx].ray.origin + ((intersection.t + 0.001f) * pathSegments[idx].ray.direction);
			//pathSegments[idx].medium = glm::dot(pathSegments[idx].ray.direction, intersection.surfaceNormal) > 0 ? intersection.mediumInterface.outside : intersection.mediumInterface.inside;

			// TODO: make work for both volume types
			pathSegments[idx].medium = insideMedium(pathSegments[idx], intersection.tMin, intersection.tMax, 0) ? intersection.mediumInterface.inside : intersection.mediumInterface.outside;

			//pathSegments[idx].remainingBounces--;
			pathSegments[idx].prev_hit_null_material = true;
			return;
		}

		Material material = materials[intersection.materialId];

		// Hit a light
		if (material.emittance > 0.0f) {
			if (pathSegments[idx].remainingBounces == gui_params.max_depth || pathSegments[idx].prev_hit_was_specular) {
				// only color lights on first hit
				pathSegments[idx].accumulatedIrradiance += (material.R * material.emittance) * pathSegments[idx].rayThroughput / pathSegments[idx].r_u;
			}
			else {
				if (glm::dot(intersection.surfaceNormal, glm::normalize(pathSegments[idx].ray.direction)) > 0.0001f) {

				}
				else {
					int light_ID = -1;
					for (int light_iter = 0; light_iter < scene_info.lights_size; light_iter++) {
						if (lights[light_iter].geom_ID == intersection.objID) {
							light_ID = light_iter;
							break;
						}
					}
					float dist = glm::length(pathSegments[idx].ray.origin - (pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction));
					float pdf_L = (intersection.t * intersection.t) / (glm::abs(glm::dot(intersection.surfaceNormal, glm::normalize(pathSegments[idx].ray.direction))) * geoms[intersection.objID].scale.x * geoms[intersection.objID].scale.y);
					pdf_L *= (1.0f / (float)scene_info.lights_size);
					if (gui_params.importance_sampling == UNI) {
						pathSegments[idx].accumulatedIrradiance += (material.R * material.emittance) * pathSegments[idx].rayThroughput;
					}
					else if (gui_params.importance_sampling == UNI_NEE_MIS) {
						pathSegments[idx].r_l *= pdf_L;
						pathSegments[idx].accumulatedIrradiance += (material.R * material.emittance) * pathSegments[idx].rayThroughput / (pathSegments[idx].r_u + pathSegments[idx].r_l);
					}
				}
			}
			pathSegments[idx].remainingBounces = 0;
			return;
		}

		pathSegments[idx].prev_hit_was_specular = material.type == SPEC_BRDF || material.type == SPEC_BTDF || material.type == SPEC_GLASS;

		if (gui_params.importance_sampling == NEE || gui_params.importance_sampling == UNI_NEE_MIS) {
		  if (!pathSegments[idx].prev_hit_was_specular) {

			glm::vec3 Ld = directLightSample(idx, false, pathSegments[idx], materials, intersection, geoms, tris,
			  media, media_density, direct_light_rays[idx], direct_light_isects[idx], lights, lbvh, gui_params, scene_info, rng, u01);

			pathSegments[idx].accumulatedIrradiance += pathSegments[idx].rayThroughput * Ld;

		  }
		}

		glm::vec3 wi = glm::vec3(0.0f);
		float pdf = 0.0f;
		float absDot = 0.0f;
		glm::vec3 f = Sample_f(material, pathSegments[idx], intersection, &wi, &pdf, absDot, rng, u01);
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
		image[iterationPath.pixelIndex] += iterationPath.accumulatedIrradiance;
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

struct medium_sort
{
	__host__ __device__
		bool operator()(const PathSegment& p0, const PathSegment& p1)
	{
		return p0.medium < p1.medium;
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

void fullVolPathtrace(uchar4* pbo, int frame, int iter, GuiParameters& gui_params, SceneInfo& scene_info) {

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

	//// Pick wavelength to sample
	//thrust::default_random_engine& rng = makeSeededRandomEngine_FullVol(iter, iter, traceDepth);
	//thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
	//int rgbWavelength = pickRGBWavelength(u01(rng));

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;

	dim3 numblocksPathSegmentTracing = (pixelcount_fullvol + blockSize1d - 1) / blockSize1d;

	// gen ray
	generateRayFromCamera_FullVol << <blocksPerGrid2d, blockSize2d >> > (cam,
		iter, traceDepth, dev_paths);

	// sorting variables
	PathSegment* dev_path_end = dev_paths + pixelcount_fullvol;
	int num_paths = dev_path_end - dev_paths;
	int compact_num_paths = num_paths;
	thrust::device_ptr<PathSegment> dev_thrust_paths = thrust::device_pointer_cast(dev_paths);
	thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections = thrust::device_pointer_cast(dev_intersections);

	while (!iterationComplete) {
		//std::cout << "DEPTH: " << depth << std::endl;
		// When intersecting with primitive, determine if there is a medium transition or not
		// Update isect struct's mediumInterface member variable with the appropriate mediumInterface
		//timer().startGpuTimer();
		computeIntersections_FullVol << <numblocksPathSegmentTracing, blockSize1d >> > (
			compact_num_paths
			, depth
			, dev_paths
			, dev_geoms
			, dev_tris
			, dev_media
			, dev_intersections
			, dev_lbvh
			, scene_info
			);
		//timer().endGpuTimer();
		//printElapsedTime(timer().getGpuElapsedTimeForPreviousOperation(), "(Compute Intersections, CUDA Measured)");

#ifdef MEDIUM_SORT
		cudaDeviceSynchronize();
#endif

		depth++;

		// Sort paths by medium type
#ifdef MEDIUM_SORT
		dev_thrust_paths = thrust::device_pointer_cast(dev_paths);
		thrust::sort_by_key(dev_thrust_paths, dev_thrust_paths + compact_num_paths, dev_intersections, medium_sort());
#endif

		// Attenuating ray throughput with medium stuff (phase function)
		// Check if throughput is black, and break out of loop (set remainingBounces to 0)
		// If medium interaction is valid, then sample light and pick new direction by sampling phase function distribution
		// Else, handle surface interaction
		//timer().startGpuTimer();
		sampleParticipatingMedium_FullVol << <numblocksPathSegmentTracing, blockSize1d >> > (
			compact_num_paths,
			depth,
			dev_paths,
			dev_materials,
			dev_intersections,
			dev_geoms,
			dev_tris,
			dev_media,
			dev_media_density,
			dev_direct_light_rays,
			dev_direct_light_isects,
			dev_lights,
			dev_lbvh,
			gui_params,
			scene_info);
		//timer().endGpuTimer();
		//printElapsedTime(timer().getGpuElapsedTimeForPreviousOperation(), "(Sample Participating Medium, CUDA Measured)");

		//timer().startGpuTimer();
		handleSurfaceInteraction_FullVol << <numblocksPathSegmentTracing, blockSize1d >> > (
			compact_num_paths,
			depth,
			dev_paths,
			dev_materials,
			dev_intersections,
			dev_geoms,
			dev_tris,
			dev_media,
			dev_media_density,
			dev_direct_light_rays,
			dev_direct_light_isects,
			dev_lights,
			dev_lbvh,
			gui_params,
			scene_info);
		//timer().endGpuTimer();
		//printElapsedTime(timer().getGpuElapsedTimeForPreviousOperation(), "(Handle Surface Interactions, CUDA Measured)");
		
		// RUSSIAN ROULETTE
		if (depth > 4)
		{
			russianRouletteKernel_FullVol << <numblocksPathSegmentTracing, blockSize1d >> > (
				iter,
				pixelcount_fullvol,
				dev_paths
				);
		}

#ifdef STREAM_COMPACTION
		thrust::device_ptr<PathSegment> dev_thrust_path_end = thrust::stable_partition(dev_thrust_paths, dev_thrust_paths + compact_num_paths, is_done());
		dev_path_end = dev_thrust_path_end.get();
		compact_num_paths = dev_path_end - dev_paths;
#endif

		if (depth == traceDepth + gui_params.depth_padding || dev_paths == dev_path_end) { iterationComplete = true; }
	}


	// Assemble this iteration and apply it to the image
	finalGather_FullVol << <numblocksPathSegmentTracing, blockSize1d >> > (pixelcount_fullvol, dev_image, dev_paths);


	if ((iter & gui_params.refresh_rate) >> gui_params.refresh_bit || iter < 2) {
		// 	// Send results to OpenGL buffer for rendering
		sendImageToPBO_FullVol << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

		// Retrieve image from GPU
		cudaMemcpy(hst_scene->state.image.data(), dev_image,
			pixelcount_fullvol * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
	}

}