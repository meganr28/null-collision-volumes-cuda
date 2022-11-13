#pragma once

#include <vector>
#include "scene.h"
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

void InitDataContainer_Vol(GuiDataContainer* guiData);
void volPathtraceInit(Scene *scene);
void volPathtraceFree();
void volPathtrace(uchar4 *pbo, int frame, int iteration);
