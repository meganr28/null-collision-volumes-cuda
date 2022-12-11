GPU-Accelerated Heterogeneous Volume Rendering with Null-Collisions
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* Nick Moon
  * [LinkedIn](https://www.linkedin.com/in/nick-moon1/), [personal website](https://nicholasmoon.github.io/)
* Megan Reddy
  * [LinkedIn](https://www.linkedin.com/in/meganr25a949125/), [personal website](https://meganr28.github.io/)
* Tested on: Windows 10, AMD Ryzen 9 5900HS with Radeon Graphics @ 3301 MHz 16GB, NVIDIA GeForce RTX 3060 Laptop GPU 6GB (Personal Computer)

### Overview

<p align="center">
  <img width="507" alt="mulit_color_cloud" src="https://user-images.githubusercontent.com/20704997/205739566-2ea7d4e6-6f81-41ea-a8db-1dccc50bc5a7.png">
<p align="center"><em>Intel Cloud rendered with the null-scattering MIS framework</em></p>

[Insert more pictures/gif/video here]

**Physically-based volume rendering** is widely used in the entertainment and scientific engineering fields for rendering phenomena such as clouds, fog, smoke, and fire. This usually involves complex lighting computations, especially for volumes that vary spatially and spectrally. Production renderers leverage multiple importance sampling (MIS) to accelerate image synthesis for rendering surfaces. MIS techniques for volumes are unbiased only for homogeneous media. Therefore, we require a new technique to perform MIS for heterogeneous media. 

The [null-scattering path integral formulation](https://cs.dartmouth.edu/wjarosz/publications/miller19null.html) (Miller et al. 2019) enables us to use MIS for any media and generalizes previous techniques such as ratio tracking, delta tracking, and spectral tracking. It analytically solves for the pdf of a light path during runtime, allowing us to combine several sampling techniques at once using MIS. Additionally, null-scattering introduces fictitious matter into the volume, which does not affect light transport, but instead allows us to "homogenize" the total density and analytically sample collisions. We implement the null-scattering formulation in **CUDA** and use **NanoVDB** for loading volumetric data. 

### Presentations

[Final Project Pitch](https://docs.google.com/presentation/d/1bVFEcVQq_lp9oRMo1wMy-prI_6DmvS1U/edit?usp=sharing&ouid=114838708762215680291&rtpof=true&sd=true)

[Milestone 1 Presentation](https://docs.google.com/presentation/d/14UCT0gwEhKlZwesXNz6KYzMYFSW_foeX/edit?usp=sharing&ouid=114838708762215680291&rtpof=true&sd=true)

[Milestone 2 Presentation](https://docs.google.com/presentation/d/1hIc8dso9Vw6BNq6eRFusN4aV4GLUBG46/edit?usp=sharing&ouid=114838708762215680291&rtpof=true&sd=true)

[Milestone 3 Presentation](https://docs.google.com/presentation/d/15A4sxapjhbVR1eCHo42OMfnLYpHewG0q/edit?usp=sharing&ouid=114838708762215680291&rtpof=true&sd=true)

### Features Implemented

- Completed
    * Heterogeneous media
      * Null-scattering MIS (next-event estimation and phase sampling)
      * Delta tracking
    * Homogeneous media
    * Interactions between surface and media
    * Volumes on the inside and outside of objects (medium interfaces)
    * Loading .vdb files
- In-Progress 
    * Debug and remove bias in null-scattering MIS renders (white pixels show up in renders)
    * Spectral MIS (with [hero wavelength sampling](https://cgg.mff.cuni.cz/publications/hero-wavelength-spectral-sampling/))
      
### Usage

#### Rendering Controls

- `Integrator`
- `Importance Sampling`
- `Max Ray Depth` - number of times light will bounce in each ray path
- `Extra Depth Padding`
- `Refresh Rate`

#### Camera Controls

- `FOV`
- `Focal Distance`
- `Lens Radius`

#### Volumetric Controls

- `Absorption` - amount of light absorbed while interacting with the medium (higher = darker)
- `Scattering` - amount of light scattering inside of the medium (out-scattering and in-scattering)
- `Asymmetry` - influences the direction of light scattering within the medium
- `Density Offset`
- `Density Scale`

### Build Instructions

To build this project, ensure that you have a **CUDA-enabled** NVIDIA GPU. We have provided some other
basic requirements below:

1. Open the `.sln` file in Visual Studio and build in **Release** mode
2. In order to run a `.txt` file from the `scenes` folder, you must provide a command line argument. You can do this two ways:
    * Call the program with the argument: `null-collision-volumes-cuda scenes/cornell_boxes.txt` 
    * In Visual Studio, navigate to your project `Properties` and select `Configuration Properties -> Debugging -> Command Arguments` and provide the path to the scene file:
 `../scenes/cornell_boxes.txt`. Note that you may need to provide the full path instead of the relative path.
3. In `Properties -> C/C++ -> Optimization`, select `Maximum Optimization (Favor Speed) (/O2)`
4. In `Properties -> C/C++ -> Code Generation -> Runtime Library`, select `Multi-threaded (/MT)`
5. When building with `cmake`, if you run into an issue where it cannot find a library file, make sure the appropriate `.lib` file is in the `external` folder.

#### OpenVDB and NanoVDB

This project depends on OpenVDB and NanoVDB for loading volumetric data. We followed the build instructions for `Windows`, however, the [official OpenVDB development repository](https://github.com/AcademySoftwareFoundation/openvdb) has directions for other platforms.
The installation process for Windows was quite complicated and lengthy for us, so we've included the steps of our process incase it is of use to anyone else. Note that these might not necessarily be executed in order, and that building OpenVDB might
differ depending on your system setup.

1. Install [vcpkg](https://github.com/microsoft/vcpkg), [CMake](https://cmake.org/), and [Visual Studio](https://visualstudio.microsoft.com/downloads/).
2. Run these commands in the directory with `vcpkg.exe` to install OpenVDB dependencies: 

```
vcpkg install zlib:x64-windows
vcpkg install blosc:x64-windows
vcpkg install tbb:x64-windows
vcpkg install boost-iostreams:x64-windows
vcpkg install boost-any:x64-windows
vcpkg install boost-algorithm:x64-windows
vcpkg install boost-uuid:x64-windows
vcpkg install boost-interprocess:x64-windows
vcpkg install openvdb:x64-windows
```

3. Clone and build the OpenVDB repository using the instructions [here](https://github.com/AcademySoftwareFoundation/openvdb) under `Windows -> Building OpenVDB`.
4. Place the resulting `openvdb` directory within your project.
5. Create a Visual Studio solution with CMake. We found the instructions [here](https://visualstudio.microsoft.com/downloads/) quite helpful (under `Build Steps`). 
5. In `FindOpenVDB.cmake`, remove lines 655-662 (this is old functionality). This file should be located under `vcpkg/installed/x64-windows/share/openvdb`.
6. In `CMakeLists.txt`, change `cmake_minimum_required(VERSION 3.1)` to `cmake_minimum_required(VERSION 3.18)`.
7. In `CMakeLists.txt`, add the following lines at the bottom of the file to include OpenVDB as a dependency. Replace the `vcpkg` path with the path to your installation.

```
list(APPEND CMAKE_MODULE_PATH "C:/src/vcpkg/vcpkg/installed/x64-windows/share/openvdb")
find_package(OpenVDB REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} OpenVDB::openvdb)
```

8. Include the appropriate header files in your project and see if you can build successfully. If not, check the OpenVDB site for [Troubleshooting tips](https://www.openvdb.org/documentation/doxygen/build.html#buildTroubleshooting). 
9. Note that NanoVDB files are included in the `openvdb` directory that you cloned. 

### Concepts

#### Null-Scattering

Include diagram with null particles here. 

#### Null-Scattering MIS

Explain unidirectional and next-event estimation. Also explain why path integral formulation gives us
an advantage over previous methods (we can calculate pdf).

### Pipeline 

Include diagram of kernel layout. 

### Results

#### Loading Volumetric Data with NanoVDB

| VDB 1  | VDB 2 | VDB 3
|:----------:    |:-------------:  |:-------------:  |
| ![](img/final/performance/uni_low_density_200iter.PNG) | ![](img/final/performance/nee_low_density_200iter.PNG) | ![](img/final/performance/mis_low_density_200iter.PNG)

#### Unidirectional, Next-Event Estimation (NEE), and Uni + NEE MIS

Case where unidirectional performs better

| Unidirectional  | Next-Event Estimation (NEE) | Unidirectional + NEE
|:----------:    |:-------------:  |:-------------:  |
| ![](img/final/performance/uni_low_density_200iter.PNG) | ![](img/final/performance/nee_low_density_200iter.PNG) | ![](img/final/performance/mis_low_density_200iter.PNG)

Case where NEE performs better

| Unidirectional  | Next-Event Estimation (NEE) | Unidirectional + NEE
|:----------:    |:-------------:  |:-------------:  |
| ![](img/final/performance/uni_high_density_200iter.PNG) | ![](img/final/performance/nee_high_density_200iter.PNG) | ![](img/final/performance/mis_high_density_200iter.PNG)

#### Varying Extinction (Absorption + Scattering) 

##### Absorption

Explain absorption and how changing it's value affects appearance.

| sigma_a = 0.1  |  sigma_a = 0.2 | sigma_a = 0.4 | sigma_a = 0.8 |
|:----------:    |:-------------:  |:-------------:  |:-------------:  |
| ![](img/final/performance/parameters_converged/absorption/02_sca_1_ab_001_asy_1_den.PNG)  |  ![](img/final/performance/parameters_converged/absorption/02_sca_2_ab_001_asy_1_den.PNG)  | ![](img/final/performance/parameters_converged/absorption/02_sca_4_ab_001_asy_1_den.PNG) | ![](img/final/performance/parameters_converged/absorption/02_sca_8_ab_001_asy_1_den.PNG)

##### Scattering

Explain scattering and how changing it's value affects appearance.

| sigma_s = 0.1  | sigma_s = 0.2 | sigma_s = 0.4 | sigma_s = 0.8 |
|:----------:    |:-------------:  |:-------------:  |:-------------:  |
| ![](img/final/performance/parameters_converged/scattering/1_sca_02_ab_001_asy_1_den.PNG)  |  ![](img/final/performance/parameters_converged/scattering/2_sca_02_ab_001_asy_1_den.PNG)  | ![](img/final/performance/parameters_converged/scattering/4_sca_02_ab_001_asy_1_den.PNG) | ![](img/final/performance/parameters_converged/scattering/8_sca_02_ab_001_asy_1_den.PNG)

##### Density

Explain density and how changing it's value affects appearance.

| 1x  | 5x | 10x | 30x |
|:----------:    |:-------------:  |:-------------:  |:-------------:  |
| ![](img/final/performance/parameters_converged/density/2_sca_02_ab_001_asy_1_den.PNG)  |  ![](img/final/performance/parameters_converged/density/2_sca_02_ab_001_asy_5_den.PNG)  | ![](img/final/performance/parameters_converged/density/2_sca_02_ab_001_asy_10_den.PNG) | ![](img/final/performance/parameters_converged/density/2_sca_02_ab_001_asy_30_den.PNG)

#### Henyey-Greenstein Phase Asymmetry 

[Talk about forward and back-scattering and what phase asymmetry (g) controls]

| Back Scattering (g = -0.8) |   Equal Scattering (g = 0.001) |   Forward Scattering (g = 0.8) |
|:----------:    |:-------------:  |:-------------:  |
| ![](img/final/performance/parameters_converged/asymmetry/2_sca_2_ab_001_asy_1_den.PNG)  |  ![](img/final/performance/parameters_converged/asymmetry/2_sca_2_ab_8_asy_1_den.PNG)  | ![](img/final/performance/parameters_converged/asymmetry/2_sca_2_ab_neg8_asy_1_den.PNG)

### Performance

#### Testing Parameters

The following performance results were obtained 

The camera and lighting parameters are the same between the two scenes; the only difference is the setting in which we placed
smoke plume. Unless otherwise noted, we used these additional parameters:

- `Absorption`: 0.02
- `Scattering`: 0.2061
- `Phase Asymmetry`: 0.001
- `Iteration Count`: 20
- `Ray Depth`: 1
- `Resolution`: 800 x 800

For testing, we turned GUI rendering off (this took about 20 ms of render time). We wrapped the call to our `fullVolPathtrace` function
with a call to our `PerformanceTimer` class's functions `startGpuTimer` and `endGpuTimer`. We recorded the average rendering time over 
20 iterations. 

#### Unidirectional, Next-Event Estimation (NEE), and Uni + NEE MIS

[Discussion here - see above pictures for results]

![](img/final/performance/graphs/mis_low_density.png)

![](img/final/performance/graphs/mis_high_density.png)

#### Varying Absorption, Scattering, and Phase Asymmetry

Figure XX below showcases the impact of varying the absorption, scattering, and
ph

![](img/final/performance/graphs/absorption.png)

Figure XX below showcases the impact of varying the absorption, scattering, and
ph

![](img/final/performance/graphs/scattering.png)

Figure XX below showcases the impact of varying the absorption, scattering, and
ph

![](img/final/performance/graphs/asymmetry.png)

#### Single Scattering vs. Multiple Scattering

Figure XX below showcases the impact of increasing the maximum allowed bounces that can 
occur within a medium when a scattering event takes place. This increases the runtime in a similar way
that increasing the maximum ray depth in a surface path tracer will, except in this
cases all of the additional depth is spent inside the same volume.

![](img/final/performance/graphs/max_depth.png)

#### Varying Max Density

Figure XX below showcases the impact of artificially augmenting the max density by some constant amount.
This means that this test doesn't change the actualy density data in the vdb grid, but instead
scales the calculated max density of the grid up. Doing this will still yield a physically correct
result, however will dramatically increase the runtime. This is because the max density
is used to calculate the majorant for the vdb volumetric grid, and this majorant is what determines
the rate of interaction sampling within the medium. Increasing this max density value will
thus increase the amount of marching through the volume the integrator is performing, because
more steps of smaller distances are being performed.

![](img/final/performance/graphs/max_density.png)

#### Varying Scene Type (Box vs. Void)

Figure XX below showcases the impact of having participating media and surfaces interact within
a scene. While the lighting benefits of the global illumination showcase a much more realistic
looking smoke cloud, the performance impact is much worse. This is because, in the scene
with nothing but a single area light, most of the rays either end when cast directly from the camera,
or end as soon as they escape the bouding box of the participating medium. However, in the
scene with the smoke surrounded on all sides by surfaces, all rays that bounce on the walls will
potentially enter the medium at subsequent bounces, and rays that leave the medium will then
go on to interact with the walls. This means that adding an additional maximum ray depth means
more multiple scattering in indirect lighting passes, which is realistic, but definitely a performance
hit. Future work could entail seperating the amount of multiple scattering within volumes from
the global illumination of the surfaces in the scene.

![](img/final/performance/graphs/box_vs_no_box.png)

#### Other Optimizations

We tried a few optimization strategies to speed up our volume renderer. One idea was to use stream compaction 
to remove terminated rays after each bounce. Another strategy was to sort the rays by their medium type, where rays
heading into a medium and those not heading into a medium will be grouped together. The intention was to ensure that 
similar rays would follow similar branches in the code, thus creating coherency and speeding up the code. Unfortunately,
neither of these strategies seemed to help and instead decreased performance. 

Some minor optimizations we made were reducing the number of global memory reads and the number of parameters being passed
into each kernel. Once we removed some of our unused parameters and consolidated some of the others, we gained a small performance boost. 
Our `sampleParticipatingMedium` and `handleSurfaceInteraction` kernels are faster by 2 ms with these changes.

### Future Work Directions

There are many interesting ways to extend this project and several features that we would like to implement in a future iteration of this work:

- Spectral rendering
- Spectral MIS
- Subsurface scattering
- Bidirectional path tracing
- Further Optimization

### References

- [A null-scattering path integral formulation of light transport](https://cs.dartmouth.edu/wjarosz/publications/miller19null.html) - Miller et al. 2019
- [Dartmouth CS 87 Rendering Algorithms Course Notes](https://cs87-dartmouth.github.io/Fall2022/schedule.html) - Wojciech Jarosz
- [Path Tracing in Production - Volumes](https://jo.dreggn.org/path-tracing-in-production/2019/christopher_kulla.pdf) - Christopher Kulla
- [PBRT v3 Chapter 15 - Volumetric Light Transport](https://www.pbr-book.org/3ed-2018/Light_Transport_II_Volume_Rendering)
- [PBRT v4](https://github.com/mmp/pbrt-v4)

#### Volumetric Data

- [Open/NanoVDB Repository](https://www.openvdb.org/) - Smoke Plumes, Cloud Bunny
- [Intel Volumetric Clouds Library](https://dpel.aswf.io/intel-cloud-library/) - Intel Cloud
- [Disney Cloud Dataset](https://disneyanimation.com/resources/clouds/) - Disney Cloud
- [JangaFX Free VDB Animations](https://jangafx.com/software/embergen/download/free-vdb-animations/)

#### Acknowledgements

We would like to thank the following individuals for their help and support throughout this project:

- Yining Karl Li
- Bailey Miller
- Wojciech Jarosz
- Adam Mally