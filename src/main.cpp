#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "main.h"
#include "preview.h"
#include <cstring>

#include <glm/gtx/string_cast.hpp>

#define DEFAULT_INTEGRATOR NULL_SCATTERING_MIS
#define DEFAULT_IMPORTANCE_SAMPLING UNI_NEE_MIS

static std::string startTimeString;

// Camera Controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

// Integrator Selection Parameters
IntegratorType ui_integrator = DEFAULT_INTEGRATOR;
IntegratorType last_integrator = DEFAULT_INTEGRATOR;
IntegratorType previous_integrator = DEFAULT_INTEGRATOR;

ImportanceSampling ui_importance_sampling = DEFAULT_IMPORTANCE_SAMPLING;
ImportanceSampling last_importance_sampling = DEFAULT_IMPORTANCE_SAMPLING;

// Path Tracing Parameters
int ui_max_ray_depth = 2;
int last_max_ray_depth = 2;
int ui_depth_padding = 2;
int last_depth_padding = 2;
int ui_refresh_bit = 0;
int last_refresh_bit = 0;
int refresh_rate = 1;

// Camera Parameters
float ui_fov = 19.5f;
float last_fov = 19.5f;
float ui_focal_distance = 17.9f;
float last_focal_distance = 17.9f;
float ui_lens_radius = 0.0f;
float last_lens_radius = 0.0f;

// Volumetric Parameters
glm::vec3 ui_sigma_a = glm::vec3(0.15f);
glm::vec3 last_sigma_a = glm::vec3(0.15f);
glm::vec3 ui_sigma_s = glm::vec3(0.15f);
glm::vec3 last_sigma_s = glm::vec3(0.15f);
float ui_g= 0.15f;
float last_g = 0.15f;
float ui_density_offset = 0.0f;
float last_density_offset = 0.0f;
float ui_density_scale = 1.0f;
float last_density_scale = 1.0f;


static bool camchanged = true;
static bool integratorchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
RenderState* renderState;
int iteration;

int cur_x;
int cur_y;

int width;
int height;

bool beginning = true;

PerformanceTimer& timer()
{
	static PerformanceTimer timer;
	return timer;
}

template<typename T>
void printElapsedTime(T time, std::string note = "")
{
	std::cout << time << std::endl;
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
	startTimeString = currentTimeString();

	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}

	const char* sceneFile = argv[1];

	// Load scene file
	GuiParameters read_scene_to_gui = { glm::vec3(0.15f), glm::vec3(0.15f), 0.15f };
	scene = new Scene(sceneFile, read_scene_to_gui);

	//std::cout << scene->lbvh.size() << std::endl;
	//ofstream myfile;
	//myfile.open("lbvh_test.txt");
	//for (int i = 0; i < scene->lbvh.size(); i++) {
	//	myfile << "Curr Node: " << scene->lbvh[i].objectId << "\n";
	//	myfile << "Left Child: " << scene->lbvh[i].left << "\n";
	//	myfile << "Right Child: " << scene->lbvh[i].right << "\n";
	//	myfile << "AABB: " << glm::to_string(scene->lbvh[i].aabb.min) << " " << glm::to_string(scene->lbvh[i].aabb.max) << "\n";
	//}
	//myfile << scene->lbvh.size() << "\n";
	//myfile.close();

	ui_max_ray_depth = scene->state.traceDepth;
	last_max_ray_depth = scene->state.traceDepth;
	ui_fov = scene->state.camera.fov.y;
	last_fov = scene->state.camera.fov.y;
	ui_focal_distance = scene->state.camera.focal_distance;
	last_focal_distance = scene->state.camera.focal_distance;
	ui_lens_radius = scene->state.camera.lens_radius;
	last_lens_radius = scene->state.camera.lens_radius;

	if (scene->media.size() > 0) {
		ui_sigma_a = read_scene_to_gui.sigma_a;
		last_sigma_a = read_scene_to_gui.sigma_a;
		ui_sigma_s = read_scene_to_gui.sigma_s;
		last_sigma_s = read_scene_to_gui.sigma_s;
		ui_g = read_scene_to_gui.g;
		last_g = read_scene_to_gui.g;
	}



	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	cur_x = 0;
	cur_y = 0;
	renderState = &scene->state;
	Camera& cam = renderState->camera;
	width = cam.resolution.x;
	height = cam.resolution.y;

	glm::vec3 view = cam.view;
	glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);

	cameraPosition = cam.position;

	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);

	// Initialize CUDA and GL components
	init();

	// GLFW main loop
	mainLoop();

	return 0;
}

void saveImage() {
	float samples = iteration;
	// output image file
	image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 pix = renderState->image[index];
			img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
		}
	}

	std::string filename = renderState->imageName;
	std::ostringstream ss;
	ss << filename << "." << startTimeString << "." << samples << "samp";
	filename = ss.str();

	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {

	if (last_importance_sampling != ui_importance_sampling) {
		last_importance_sampling = ui_importance_sampling;
		camchanged = true;

	}

	if (last_max_ray_depth != ui_max_ray_depth) {
		last_max_ray_depth = ui_max_ray_depth;
		camchanged = true;
		
	}

	if (last_depth_padding != ui_depth_padding) {
		last_depth_padding = ui_depth_padding;
		camchanged = true;

	}
	if (last_refresh_bit != ui_refresh_bit) {
		last_refresh_bit = ui_refresh_bit;
		refresh_rate = 1 << ui_refresh_bit;
		camchanged = true;
	}



	if (last_fov != ui_fov) {
		last_fov = ui_fov;
		camchanged = true;
	}
	if (last_focal_distance != ui_focal_distance) {
		last_focal_distance = ui_focal_distance;
		camchanged = true;
	}
	if (last_lens_radius != ui_lens_radius) {
		last_lens_radius = ui_lens_radius;
		camchanged = true;
	}



	if (last_sigma_a != ui_sigma_a) {
		last_sigma_a = ui_sigma_a;
		camchanged = true;
	}
	if (last_sigma_s != ui_sigma_s) {
		last_sigma_s = ui_sigma_s;
		camchanged = true;
	}
	if (last_g != ui_g) {
		last_g = ui_g;
		camchanged = true;
	}

	if (last_density_offset != ui_density_offset) {
		last_density_offset = ui_density_offset;
		camchanged = true;
	}

	if (last_density_scale != ui_density_scale) {
		last_density_scale = ui_density_scale;
		camchanged = true;
	}

	if (last_integrator != ui_integrator) {
		previous_integrator = last_integrator;
		last_integrator = ui_integrator;
		integratorchanged = true;
		camchanged = true;

	}


	if (camchanged) {
		iteration = 0;
		Camera& cam = renderState->camera;
		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.y = zoom * cos(theta);
		cameraPosition.z = zoom * cos(phi) * sin(theta);

		cam.view = -glm::normalize(cameraPosition);
		glm::vec3 v = cam.view;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		cam.up = glm::cross(r, v);
		cam.right = r;

		cam.position = cameraPosition;
		cameraPosition += cam.lookAt;
		cam.position = cameraPosition;
		camchanged = false;


		scene->state.traceDepth = ui_max_ray_depth;
		float yscaled = tan(ui_fov * (PI / 180));
		float xscaled = (yscaled * scene->state.camera.resolution.x) / scene->state.camera.resolution.y;
		float fovx = (atan(xscaled) * 180) / PI;
		scene->state.camera.fov = glm::vec2(fovx, ui_fov);

		scene->state.camera.pixelLength = glm::vec2(2 * xscaled / (float)scene->state.camera.resolution.x,
			2 * yscaled / (float)scene->state.camera.resolution.y);

		scene->state.camera.focal_distance = ui_focal_distance;
		scene->state.camera.lens_radius = ui_lens_radius;

	}

	if (integratorchanged) {
		if (previous_integrator == NULL_SCATTERING_MIS) {
			if (!beginning) {
				//ui_sigma_a.x *= 0.33f;
				//ui_sigma_s.x *= 0.33f;
			}
			
			fullVolPathtraceFree();
		}
		else if (previous_integrator == DELTA_TRACKING_NEE) {
			if (!beginning) {
				//ui_sigma_a.x *= 3.0f;
				//ui_sigma_s.x *= 3.0f;
			}
			
			volPathtraceFree();
		}
		else if (previous_integrator == SURFACE_ONLY_MIS) {
			pathtraceFree();
		}

		beginning = false;

		if (ui_integrator == NULL_SCATTERING_MIS) {

			fullVolPathtraceInit(scene);
		}
		else if (ui_integrator == DELTA_TRACKING_NEE) {

			volPathtraceInit(scene);
		}
		else if (ui_integrator == SURFACE_ONLY_MIS) {

			pathtraceInit(scene);
		}
		integratorchanged = false;
	}

	
	//GuiParameters gui_params = { glm::vec3(ui_sigma_a), glm::vec3(ui_sigma_s), ui_g };
	GuiParameters gui_params = { glm::vec3(ui_sigma_a.x, ui_sigma_a.x, ui_sigma_a.x), 
		glm::vec3(ui_sigma_s.x, ui_sigma_s.x, ui_sigma_s.x), 
		ui_g, 
		ui_max_ray_depth,
		ui_depth_padding,
		refresh_rate,
		ui_refresh_bit,
		ui_density_offset,
		ui_density_scale,
		ui_importance_sampling };

	// Initialize scene info
	SceneInfo scene_info = { scene->geoms.size(), scene->media.size(), scene->lights.size() };

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	if (iteration == 0) {
		if (ui_integrator == NULL_SCATTERING_MIS) {
			fullVolResetImage();
		}
		else if (ui_integrator == DELTA_TRACKING_NEE) {
			volResetImage();
		}
		else if (ui_integrator == SURFACE_ONLY_MIS) {
			resetImage();
		}
	}

	if (iteration < renderState->iterations) {
		uchar4* pbo_dptr = NULL;
		iteration++;
		cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

		// execute the kernel
		int frame = 0;

		if (ui_integrator == NULL_SCATTERING_MIS) {
			timer().startGpuTimer();
			fullVolPathtrace(pbo_dptr, frame, iteration, gui_params, scene_info);
			timer().endGpuTimer();
			printElapsedTime(timer().getGpuElapsedTimeForPreviousOperation());
		}
		else if (ui_integrator == DELTA_TRACKING_NEE) {
			volPathtrace(pbo_dptr, frame, iteration, gui_params);
		}
		else if (ui_integrator == SURFACE_ONLY_MIS) {
			pathtrace(pbo_dptr, frame, iteration);
		}

		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
	}
	/*lse {
		saveImage();
		pathtraceFree();
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}*/
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			saveImage();
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_S:
			saveImage();
			break;
		case GLFW_KEY_SPACE:
			camchanged = true;
			renderState = &scene->state;
			Camera& cam = renderState->camera;
			cam.lookAt = ogLookAt;
			break;
		}
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (MouseOverImGuiWindow())
	{
		return;
	}
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
	if (leftMousePressed) {
		// compute new camera parameters
		phi -= ((xpos - lastX) / width) * 2.5f;
		theta -= ((ypos - lastY) / height) * 2.5f;
		theta = std::fmax(0.001f, std::fmin(theta, PI));
		camchanged = true;
	}
	else if (rightMousePressed) {
		renderState = &scene->state;
		Camera& cam = renderState->camera;
		zoom += ((ypos - lastY) / height) * 7.5f * glm::length(cam.position);
		zoom = std::fmax(0.1f * glm::length(cam.position), zoom);
		camchanged = true;
	}
	else if (middleMousePressed) {
		renderState = &scene->state;
		Camera& cam = renderState->camera;
		glm::vec3 forward = cam.view;
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		glm::vec3 right = cam.right;
		right.y = 0.0f;
		right = glm::normalize(right);

		cam.lookAt -= (float)(xpos - lastX) * right * 0.01f * glm::length(cam.position) * 0.1f;
		cam.lookAt += (float)(ypos - lastY) * forward * 0.01f * glm::length(cam.position) * 0.1f;
		camchanged = true;
	}
	lastX = xpos;
	lastY = ypos;
}
