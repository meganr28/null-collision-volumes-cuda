#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "main.h"
#include "preview.h"
#include <cstring>

//#define PATH_INTEGRATOR
//#define VOLUME_INTEGRATOR
//#define FULL_VOLUME_INTEGRATOR

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;


IntegratorType ui_integrator = NULL_SCATTERING_MIS;
IntegratorType last_integrator = NULL_SCATTERING_MIS;
IntegratorType previous_integrator = NULL_SCATTERING_MIS;

int ui_max_ray_depth = 8;
int last_max_ray_depth = 8;
glm::vec3 ui_sigma_a = glm::vec3(0.15f);
glm::vec3 last_sigma_a = glm::vec3(0.15f);
glm::vec3 ui_sigma_s = glm::vec3(0.15f);
glm::vec3 last_sigma_s = glm::vec3(0.15f);
float ui_g= 0.15f;
float last_g = 0.15f;


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
	ui_max_ray_depth = scene->state.traceDepth;
	last_max_ray_depth = scene->state.traceDepth;

	if (scene->media.size() > 0) {
		ui_sigma_a = read_scene_to_gui.sigma_a;
		last_sigma_a = read_scene_to_gui.sigma_a;
		ui_sigma_s = read_scene_to_gui.sigma_s;
		last_sigma_s = read_scene_to_gui.sigma_s;
		ui_g = read_scene_to_gui.g;
		last_g = read_scene_to_gui.g;
	}

	std::cout << glm::length(glm::vec3(0.02, 0.03, 0.01)) << std::endl;

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


	if (last_max_ray_depth != ui_max_ray_depth) {
		last_max_ray_depth = ui_max_ray_depth;
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

	}

	if (integratorchanged) {
		if (previous_integrator == NULL_SCATTERING_MIS) {
			if (!beginning) {
				ui_sigma_a.x *= 0.33f;
				ui_sigma_s.x *= 0.33f;
			}
			
			fullVolPathtraceFree();
		}
		else if (previous_integrator == DELTA_TRACKING_NEE) {
			if (!beginning) {
				ui_sigma_a.x *= 3.0f;
				ui_sigma_s.x *= 3.0f;
			}
			
			volPathtraceFree();
		}

		beginning = false;

		if (ui_integrator == NULL_SCATTERING_MIS) {

			fullVolPathtraceInit(scene);
		}
		else if (ui_integrator == DELTA_TRACKING_NEE) {

			volPathtraceInit(scene);
		}
		integratorchanged = false;
	}

	
	//GuiParameters gui_params = { glm::vec3(ui_sigma_a), glm::vec3(ui_sigma_s), ui_g };
	GuiParameters gui_params = { glm::vec3(ui_sigma_a.x, ui_sigma_a.x, ui_sigma_a.x), glm::vec3(ui_sigma_s.x, ui_sigma_s.x, ui_sigma_s.x), ui_g };
	
	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
	if (iteration == 0) {
		if (ui_integrator == NULL_SCATTERING_MIS) {
			fullVolResetImage();
		}
		else if (ui_integrator == DELTA_TRACKING_NEE) {
			volResetImage();
		}

#ifdef PATH_INTEGRATOR
		pathtraceFree();
		pathtraceInit(scene);
#endif
	}

	if (iteration < renderState->iterations) {
		uchar4* pbo_dptr = NULL;
		iteration++;
		cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

		// execute the kernel
		int frame = 0;
#ifdef PATH_INTEGRATOR
		pathtrace(pbo_dptr, frame, iteration);
#endif
		if (ui_integrator == NULL_SCATTERING_MIS) {
			fullVolPathtrace(pbo_dptr, frame, iteration, gui_params);
		}
		else if (ui_integrator == DELTA_TRACKING_NEE) {
			volPathtrace(pbo_dptr, frame, iteration, gui_params);
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
