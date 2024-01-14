#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
#include <CL/cl2.hpp>
#include "color.hh"
#include "ray.hh"
#include "scene.hh"
#include "theora.hh"
#include "vector.hh"
#include "random.hh"

using uniform_distribution = std::uniform_real_distribution<float>;
using color = Color<float>;
using object_ptr = std::unique_ptr<Object>;
using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

vec trace(ray r, const Object_group &objects)
{
    float factor = 1;
    const int max_depth = 50;
    int depth = 0;
    for (; depth < max_depth; ++depth)
    {
        if (Hit hit = objects.hit(r, 1e-3f, std::numeric_limits<float>::max()))
        {
            r = ray(hit.point, hit.normal + random_in_unit_sphere()); // scatter
            factor *= 0.5f;                                           // diffuse 50% of light, scatter the remaining
        }
        else
        {
            break;
        }
    }
    // if (depth == max_depth) { return vec{}; }
    //  nothing was hit
    //  represent sky as linear gradient in Y dimension
    float t = 0.5f * (unit(r.direction())(1) + 1.0f);
    return factor * ((1.0f - t) * vec(1.0f, 1.0f, 1.0f) + t * vec(0.5f, 0.7f, 1.0f));
}

void print_column_names(const char *version)
{
    std::cout << std::setw(20) << "Time step";
    std::cout << std::setw(20) << "No. of steps";
    std::cout << std::setw(20) << version << " time";
    std::cout << '\n';
}

void ray_tracing_cpu()
{
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    using std::chrono::seconds;
    int nx = 600, ny = 400, nrays = 100;
    Pixel_matrix<float> pixels(nx, ny);
    thx::screen_recorder recorder("out-cpu.ogv", nx, ny);
    Object_group objects;
    objects.add(object_ptr(new Sphere(vec(0.f, 0.f, -1.f), 0.5f)));
    objects.add(object_ptr(new Sphere(vec(0.f, -1000.5f, -1.f), 1000.f)));
    Camera camera;
    uniform_distribution distribution(0.f, 1.f);
    float gamma = 2;
    const int max_time_step = 60;
    print_column_names("OpenMP");
    duration total_time = duration::zero();
    for (int time_step = 1; time_step <= max_time_step; ++time_step)
    {
        auto t0 = clock_type::now();
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
        for (int j = 0; j < ny; ++j)
        {
            for (int i = 0; i < nx; ++i)
            {
                vec sum;
                for (int k = 0; k < nrays; ++k)
                {
                    float u = (i + distribution(prng)) / nx;
                    float v = (j + distribution(prng)) / ny;
                    sum += trace(camera.make_ray(u, v), objects);
                }
                sum /= float(nrays);         // antialiasing
                sum = pow(sum, 1.f / gamma); // gamma correction
                pixels(i, j) = to_color(sum);
            }
        }
        auto t1 = clock_type::now();
        const auto dt = duration_cast<microseconds>(t1 - t0);
        total_time += dt;
        std::clog
            << std::setw(20) << time_step
            << std::setw(20) << max_time_step
            << std::setw(20) << dt.count()
            << std::endl;
        std::ofstream out("out-cpu.ppm");
        out << pixels;
        recorder.record_frame(pixels);
        camera.move(vec{0.f, 0.f, 0.1f});
    }
    std::clog << "Ray-tracing time: " << duration_cast<seconds>(total_time).count()
              << "s" << std::endl;
    std::clog << "Movie time: " << max_time_step / 60.f << "s" << std::endl;
}

struct OpenCL
{
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void ray_tracing_gpu()
{
    /***** initial setup *****/
    // find OpenCL platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
    {
        std::cerr << "Unable to find OpenCL platforms\n";
        return;
    }
    cl::Platform platform = platforms[0];
    std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
    // create context
    cl_context_properties properties[] =
        {CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);
    // get all devices associated with the context
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::Device device = devices[0];
    std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
    std::ifstream ifs("opencl_source.cl");
    std::string src((std::istreambuf_iterator<char>(ifs)),
                    (std::istreambuf_iterator<char>()));
    cl::Program program(context, src);
    // compile the programme
    program.build(devices);
    cl::CommandQueue queue(context, device);
    OpenCL opencl{platform, device, context, program, queue};
    /***** initial setup *****/

    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    using std::chrono::seconds;
    int nx = 600, ny = 400, nrays = 100;
    Pixel_matrix<float> pixels(nx, ny);
    thx::screen_recorder recorder("out-gpu.ogv", nx, ny);
    std::vector<Sphere> spheres = {
        Sphere{vec(0.f, 0.f, -1.f), 0.5f},
        Sphere{vec(0.f, -1000.5f, -1.f), 1000.f}};

    uniform_distribution distribution(0.f, 1.f);
    float gamma = 2;
    const int maxTimeStep = 60;
    print_column_names("OpenCL");

    float cameraOrigin[4] = {13.f, 2.f, 3.f, 0};
    float cameraMoveDirection[4] = {0.f, 0.f, 0.1f, 0};
    float cameraLLCorner[4] = {3.02374f, -1.22628f, 3.4122f, 0};
    float cameraHorizontal[4] = {1.18946f, 0.f, -5.15434f, 0};
    float cameraVertical[4] = {-0.509421f, 3.48757f, -0.117559f, 0};

    std::vector<float> spheresAsVector;
    for (int i = 0; i < spheres.size(); i++)
    {
        spheresAsVector.push_back(spheres[i].origin()(0));
        spheresAsVector.push_back(spheres[i].origin()(1));
        spheresAsVector.push_back(spheres[i].origin()(2));
        spheresAsVector.push_back(spheres[i].radius());
    }

    cl::Buffer cameraOriginBuffer(opencl.queue, cameraOrigin, cameraOrigin + 4, false);
    cl::Buffer cameraMoveDirectionBuffer(opencl.queue, cameraMoveDirection, cameraMoveDirection + 4, true);
    cl::Buffer cameraLLCornerBuffer(opencl.queue, cameraLLCorner, cameraLLCorner + 4, true);
    cl::Buffer cameraHorizontalBuffer(opencl.queue, cameraHorizontal, cameraHorizontal + 4, true);
    cl::Buffer cameraVerticalBuffer(opencl.queue, cameraVertical, cameraVertical + 4, true);
    cl::Buffer spheresBuffer(opencl.queue, std::begin(spheresAsVector), std::end(spheresAsVector), true);
    cl::Buffer resultBuffer(opencl.context, CL_MEM_READ_WRITE, nx * ny * 3 * sizeof(float));

    std::normal_distribution<float> dist(0.f, 1.f);
    int distr_size = 1 << 24;
    std::vector<float> distributionVector;

    for (int i = 0; i < distr_size; i++)
    {
        distributionVector.push_back(dist(prng));
    }

    cl::Buffer d_distr(opencl.queue, begin(distributionVector), end(distributionVector), true);

    cl::Kernel rayTraceKernel(opencl.program, "ray_trace");

    rayTraceKernel.setArg(0, cameraOriginBuffer);
    rayTraceKernel.setArg(1, cameraLLCornerBuffer);
    rayTraceKernel.setArg(2, cameraHorizontalBuffer);
    rayTraceKernel.setArg(3, cameraVerticalBuffer);
    rayTraceKernel.setArg(4, spheresBuffer);
    rayTraceKernel.setArg(5, 2);
    rayTraceKernel.setArg(6, d_distr);
    rayTraceKernel.setArg(7, distr_size);
    rayTraceKernel.setArg(8, resultBuffer);
    rayTraceKernel.setArg(9, ny);
    rayTraceKernel.setArg(10, nx);
    rayTraceKernel.setArg(11, nrays);
    rayTraceKernel.setArg(12, gamma);

    cl::Kernel moveCameraKernel(opencl.program, "move_camera");

    moveCameraKernel.setArg(0, cameraOriginBuffer);
    moveCameraKernel.setArg(1, cameraMoveDirectionBuffer);

    opencl.queue.flush();

    duration totalTime = duration::zero();
    for (int time_step = 1; time_step <= maxTimeStep; ++time_step)
    {
        auto t0 = clock_type::now();
        opencl.queue.enqueueNDRangeKernel(rayTraceKernel, cl::NullRange, cl::NDRange(ny, nx), cl::NullRange);
        opencl.queue.flush();

        opencl.queue.enqueueNDRangeKernel(moveCameraKernel, cl::NullRange, cl::NDRange(1), cl::NullRange);
        opencl.queue.flush();

        opencl.queue.enqueueReadBuffer(resultBuffer, true, 0, 3 * nx * ny * sizeof(float), (float *)(pixels.pixels().data()));
        opencl.queue.finish();
        auto t1 = clock_type::now();
        const auto dt = duration_cast<microseconds>(t1 - t0);
        totalTime += dt;
        std::clog
            << std::setw(20) << time_step
            << std::setw(20) << maxTimeStep
            << std::setw(20) << dt.count()
            << std::endl;
        std::ofstream out("out-gpu.ppm");
        out << pixels;
        recorder.record_frame(pixels);
    }
    std::clog << "Ray-tracing time: " << duration_cast<seconds>(totalTime).count()
              << " s" << std::endl;
    std::clog << "Movie time: " << maxTimeStep / 60.f << " s" << std::endl;
}

int main(int argc, char *argv[])
{
    ray_tracing_cpu();
    ray_tracing_gpu();
    return 0;
}
