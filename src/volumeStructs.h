#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#include <thrust/random.h>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

// Represents the Henyey-Greenstein phase function
// TODO: Maybe generalize this to more phase functions later
struct PhaseFunction {
    glm::vec3 p(const glm::vec3 &wo, const glm::vec3 &wi) {}
    glm::vec3 Sample_p(const glm::vec3 &wo, glm::vec3 *wi, const glm::vec2 &u) {}
    const float g;
};

// Represents a homogeneous medium
struct HomogeneousMedium {
    HomogeneousMedium() : sigma_a(glm::vec3(0.0)), sigma_s(glm::vec3(0.0)), sigma_t(glm::vec3(0.0)), g(0.0) {}
    HomogeneousMedium(const glm::vec3& sigma_a, const glm::vec3& sigma_s, float g)
        : sigma_a(sigma_a), sigma_s(sigma_s), sigma_t(sigma_s + sigma_a), g(g) {}
    glm::vec3 Tr(const Ray &ray) {}
    glm::vec3 Sample(const Ray &ray) {}
    glm::vec3 sigma_a; // Absorption coefficient
    glm::vec3 sigma_s; // Scattering coefficient
    glm::vec3 sigma_t; // Extinction
    float g;           // Asymmetry factor for Henyey-Greenstein
};

// Represents possible transition between two mediums
struct MediumInterface {
    MediumInterface() : inside(nullptr), outside(nullptr) {}
    MediumInterface(const HomogeneousMedium *medium) : inside(medium), outside(medium) {}
    MediumInterface(const HomogeneousMedium *inside, HomogeneousMedium *outside) : inside(inside), outside(outside) {}
    bool IsMediumTransition() const { return inside != outside;  }
    const HomogeneousMedium *inside, *outside;
};
