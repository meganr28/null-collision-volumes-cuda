#include <iostream>
#include <string>
#include <stb_image_write.h>
#include <stb_image.h>

#include "image.h"

image::image(int x, int y) :
        xSize(x),
        ySize(y),
        pixels(new glm::vec3[x * y]) {
}

image::image(std::string file_name) {
    std::cout << file_name << std::endl;
    int n;
    float* data = stbi_loadf(file_name.c_str(), &xSize, &ySize, &n, 0);
    std::cout << xSize << " " << ySize << " " << n << std::endl;

    for (int y = 0; y < ySize; y++) {
        for (int x = 0; x < xSize; x++) {
            int i = y * xSize + x;

            //std::cout << data[3 * i + 0] << " " << data[3 * i + 1] << " " << data[3 * i + 2] << std::endl;;

        }
    }



    pixels = new glm::vec3[xSize * ySize];

    for (int y = 0; y < ySize; y++) {
        for (int x = 0; x < xSize; x++) {
            int i = y * xSize + x;
            pixels[i] = glm::vec3(data[3 * i + 0], data[3 * i + 1], data[3 * i + 2]);
        }
    }

    for (int y = 0; y < ySize; y++) {
        for (int x = 0; x < xSize; x++) {
            int i = y * xSize + x;
            //pixels[i] = glm::vec3(data[3 * i + 0], data[3 * i + 1], data[3 * i + 2]);
            //std::cout << pixels[i].x << " " << pixels[i].y << " " << pixels[i].z << std::endl;;
        }
    }


    //memcpy(pixels, data, xSize * ySize * n);
}

image::~image() {
    delete pixels;
}

void image::setPixel(int x, int y, const glm::vec3 &pixel) {
    assert(x >= 0 && y >= 0 && x < xSize && y < ySize);
    pixels[(y * xSize) + x] = pixel;
}

void image::savePNG(const std::string &baseFilename) {
    unsigned char *bytes = new unsigned char[3 * xSize * ySize];
    for (int y = 0; y < ySize; y++) {
        for (int x = 0; x < xSize; x++) { 
            int i = y * xSize + x;
            glm::vec3 pix = glm::clamp(pixels[i], glm::vec3(), glm::vec3(1)) * 255.f;
            bytes[3 * i + 0] = (unsigned char) pix.x;
            bytes[3 * i + 1] = (unsigned char) pix.y;
            bytes[3 * i + 2] = (unsigned char) pix.z;
        }
    }

    std::string filename = baseFilename + ".png";
    stbi_write_png(filename.c_str(), xSize, ySize, 3, bytes, xSize * 3);
    std::cout << "Saved " << filename << "." << std::endl;

    delete[] bytes;
}

void image::saveHDR(const std::string &baseFilename) {
    std::string filename = baseFilename + ".hdr";
    stbi_write_hdr(filename.c_str(), xSize, ySize, 3, (const float *) pixels);
    std::cout << "Saved " + filename + "." << std::endl;
}
