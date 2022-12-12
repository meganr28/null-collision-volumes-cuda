#pragma once

#include <glm/glm.hpp>

using namespace std;

class image {


public:
    image(int x, int y);
    image(std::string file_name);
    ~image();
    void setPixel(int x, int y, const glm::vec3 &pixel);
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);


    int xSize;
    int ySize;
    glm::vec3* pixels;

};