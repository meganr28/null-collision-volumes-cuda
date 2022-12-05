#pragma once

#include <vector>
#include "scene.h"
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

void InitDataContainer_FullVol(GuiDataContainer* guiData);
void fullVolPathtraceInit(Scene *scene);
void fullVolResetImage();
void fullVolPathtraceFree();
void fullVolPathtrace(uchar4 *pbo, int frame, int iteration, GuiParameters& gui_params);
