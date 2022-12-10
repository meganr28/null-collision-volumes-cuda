#pragma once

#include <vector>
#include "scene.h"
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

void pathtraceInit(Scene *scene);
void pathtraceFree();
void resetImage();
void pathtrace(uchar4 *pbo, int frame, int iteration);

void pathtraceInit_Single(Scene* scene);
void pathtraceFree_Single();
void pathtrace_Single(uchar4* pbo, int frame, int x, int y);
