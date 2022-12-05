#pragma once

#include <vector>
#include "scene.h"
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

void volPathtraceInit(Scene *scene);
void volPathtraceFree();
void volResetImage();
void volPathtrace(uchar4 *pbo, int frame, int iteration, GuiParameters& gui_params);
