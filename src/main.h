#pragma once



#include <GL/glew.h>
#include <GLFW/glfw3.h>



#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "glslUtility.hpp"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>

#include "sceneStructs.h"
#include "image.h"
#include "pathtrace.h"
#include "volpathtrace.h"
#include "fullvolpathtrace.h"
#include "utilities.h"
#include "scene.h"

using namespace std;

//-------------------------------
//----------PATH TRACER----------
//-------------------------------

extern Scene* scene;
extern int iteration;

extern int width;
extern int height;

extern IntegratorType ui_integrator;
extern ImportanceSampling ui_importance_sampling;

extern int ui_max_ray_depth;
extern int ui_depth_padding;
extern int ui_refresh_bit;

extern float ui_fov;
extern float ui_focal_distance;
extern float ui_lens_radius;

extern glm::vec3 ui_sigma_a;
extern glm::vec3 ui_sigma_s;
extern float ui_g;
extern float ui_density_offset;
extern float ui_density_scale;

void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
