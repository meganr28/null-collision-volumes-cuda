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
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Tri* dev_tris = NULL;
static Light* dev_lights = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
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

void InitDataContainer_FullVol(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

// TODO: remove these when done testing
__global__ void grid_test_kernel_FullVol(const nanovdb::NanoGrid<float>* deviceGrid)
{
	if (threadIdx.x > 6)
		return;
	int i = 97 + threadIdx.x;
	auto gpuAcc = deviceGrid->getAccessor();
	printf("(%3i,0,0) NanoVDB gpu: % -4.2f\n", i, gpuAcc.getValue(nanovdb::Coord(i, i, i)));
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

	cudaMalloc(&dev_tris, scene->num_tris * sizeof(Tri));
	cudaMemcpy(dev_tris, scene->mesh_tris_sorted.data(), scene->num_tris * sizeof(Tri), cudaMemcpyHostToDevice);

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
	, BVHNode_GPU* bvh_nodes
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

#ifdef ENABLE_TRIS
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
				t1 = (cur_node.AABB_min.x - r.origin.x) * r.direction_inv.x;
				t2 = (cur_node.AABB_max.x - r.origin.x) * r.direction_inv.x;

				tmin = glm::min(t1, t2);
				tmax = glm::max(t1, t2);

				t1 = (cur_node.AABB_min.y - r.origin.y) * r.direction_inv.y;
				t2 = (cur_node.AABB_max.y - r.origin.y) * r.direction_inv.y;

				tmin = glm::max(tmin, glm::min(t1, t2));
				tmax = glm::min(tmax, glm::max(t1, t2));

				t1 = (cur_node.AABB_min.z - r.origin.z) * r.direction_inv.z;
				t2 = (cur_node.AABB_max.z - r.origin.z) * r.direction_inv.z;

				tmin = glm::max(tmin, glm::min(t1, t2));
				tmax = glm::min(tmax, glm::max(t1, t2));

				if (tmax >= tmin) {
					// we intersected AABB
					if (cur_node.tri_index != -1) {
						// this is leaf node
						// triangle intersection test
						Tri tri = tris[cur_node.tri_index];

						t = glm::dot(tri.plane_normal, (tri.p0 - r.origin)) / glm::dot(tri.plane_normal, r.direction);
						if (t >= -0.0001f) {
							P = r.origin + t * r.direction;

							// barycentric coords
							s = glm::vec3(glm::length(glm::cross(P - tri.p1, P - tri.p2)),
								glm::length(glm::cross(P - tri.p2, P - tri.p0)),
								glm::length(glm::cross(P - tri.p0, P - tri.p1))) / tri.S;

							if (s.x >= -0.0001f && s.x <= 1.0001f && s.y >= -0.0001f && s.y <= 1.0001f &&
								s.z >= -0.0001f && s.z <= 1.0001f && (s.x + s.y + s.z <= 1.0001f) && (s.x + s.y + s.z >= -0.0001f) && isect.t > t) {
								isect.t = t;
								isect.materialId = tri.mat_ID;
								isect.surfaceNormal = glm::normalize(s.x * tri.n0 + s.y * tri.n1 + s.z * tri.n2);

								// Check if surface is medium transition
								if (IsMediumTransition(tri.mediumInterface)) {
									isect.mediumInterface = tri.mediumInterface;
								}
								else {
									MediumInterface mediumInterface;
									mediumInterface.inside = pathSegments[path_index].medium;
									mediumInterface.outside = pathSegments[path_index].medium;
									isect.mediumInterface = mediumInterface;
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
#endif

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
			else {
#ifdef ENABLE_RECTS
			t = boxIntersectionTest(geom, r, tmp_normal);
#endif
			}

			if (depth == 0 && glm::dot(tmp_normal, r.direction) > 0.0) { 
				continue; 
			}
			else if (isect.t > t) {
				isect.t = t;
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
	BVHNode_GPU* bvh_nodes,
	const nanovdb::NanoGrid<float>* media_density)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		if (pathSegments[idx].remainingBounces == 0) {
			return;
		}

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
		if (rayMediumIndex >= 0) {
			if (media[rayMediumIndex].type == HOMOGENEOUS) {
				pathSegments[idx].rayThroughput *= Sample_homogeneous(media[rayMediumIndex], pathSegments[idx], intersections[idx], &mi, rayMediumIndex, u01(rng));
			}
			else {
				//pathSegments[idx].rayThroughput *= Sample_heterogeneous(media[rayMediumIndex], pathSegments[idx], intersections[idx], &mi, media_density, rayMediumIndex, rng, u01);
				glm::vec3 Tr = Sample_channel(max_depth, media[rayMediumIndex], pathSegments[idx], intersections[idx], &mi, media_density, rayMediumIndex, rng, u01);
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
			scattered = handleMediumInteraction(idx, max_depth, pathSegments, materials, intersections[idx], mi, geoms, geoms_size, tris, tris_size,
				media, media_size, media_density, direct_light_rays, direct_light_isects, lights, num_lights, bvh_nodes, rng, u01);
		}

		if (pathSegments[idx].remainingBounces == 0) {
			return;
		}

		if (scattered) {
			return;
		}

		/*pathSegments[idx].accumulatedIrradiance += pathSegments[idx].rayThroughput
			* directLightSample(idx, pathSegments, materials, intersections[idx], geoms, geoms_size, tris, tris_size,
				media, media_size, media_density, direct_light_rays, direct_light_isects, lights, num_lights, bvh_nodes, rng, u01); // TODO: * uniform sample one light;*/

		// Handle surface interaction
		ShadeableIntersection intersection = intersections[idx];

		if (intersections[idx].mi.medium == -1) {
			if (intersection.materialId < 0) {
				// Change ray direction
				pathSegments[idx].ray.origin = pathSegments[idx].ray.origin + (intersection.t * pathSegments[idx].ray.direction) + (0.001f * pathSegments[idx].ray.direction);
				pathSegments[idx].medium = glm::dot(pathSegments[idx].ray.direction, intersection.surfaceNormal) > 0 ? intersection.mediumInterface.outside :
					intersection.mediumInterface.inside;
				pathSegments[idx].remainingBounces--;
				pathSegments[idx].prev_hit_null_material = true;
				return;
			}
		}

		Material material = materials[intersection.materialId];

		if (intersections[idx].mi.medium == -1) {
			if (material.emittance > 0.0f) {
				//if (pathSegments[idx].remainingBounces == max_depth || pathSegments[idx].prev_hit_was_specular) {
					// only color lights on first hit
					pathSegments[idx].accumulatedIrradiance += (material.R * material.emittance) * pathSegments[idx].rayThroughput;
				//}
				pathSegments[idx].remainingBounces = 0;
				return;
			}

			pathSegments[idx].prev_hit_was_specular = material.type == SPEC_BRDF || material.type == SPEC_BTDF || material.type == SPEC_GLASS;

			if (pathSegments[idx].prev_hit_was_specular) {
				return;
			}
		}
	}
}

__global__ void russianRouletteKernel_FullVol(int iter, int num_paths, PathSegment* pathSegments)
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

void fullVolPathtrace(uchar4* pbo, int frame, int iter) {

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
			dev_bvh_nodes,
			dev_media_density);
		
		// RUSSIAN ROULETTE
		/*if (depth > 4)
		{
			russianRouletteKernel_FullVol << <numblocksPathSegmentTracing, blockSize1d >> > (
				iter,
				pixelcount_fullvol,
				dev_paths
				);
		}*/

		if (depth == traceDepth) { iterationComplete = true; }

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
		}
	}


	// Assemble this iteration and apply it to the image
	finalGather_FullVol << <numblocksPathSegmentTracing, blockSize1d >> > (pixelcount_fullvol, dev_image, dev_paths);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO_FullVol << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount_fullvol * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
}