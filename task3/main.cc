#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

#include "filter.hh"
#include "linear-algebra.hh"
#include "reduce-scan.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1)
{
    using namespace std::chrono;
    const auto dt = duration_cast<microseconds>(t1 - t0).count();
    if (dt == 0)
    {
        return 0;
    }
    return ((n + n + n) * sizeof(float) * 1e-9) / (dt * 1e-6);
}

void print(const char *name, std::array<duration, 5> dt)
{
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i = 0; i < 5; ++i)
    {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
        std::cout << std::setw(20) << tmp.str();
    }
    std::cout << '\n';
}

void print_column_names()
{
    std::cout << std::setw(19) << "function";
    std::cout << std::setw(20) << "OpenMP";
    std::cout << std::setw(20) << "OpenCL total";
    std::cout << std::setw(20) << "OpenCL copy-in";
    std::cout << std::setw(20) << "OpenCL kernel";
    std::cout << std::setw(20) << "OpenCL copy-out";
    std::cout << '\n';
}

struct OpenCL
{
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void profile_filter(int n, OpenCL &opencl)
{
    auto input = random_std_vector<float>(n);
    std::vector<float> result(n + 1), expected_result;
    std::vector<int> indices(n);
    auto t0 = clock_type::now();
    filter(input, expected_result, [](float x)
           { return x > 0; }); // filter positive numbers
    auto t1 = clock_type::now();
    cl::Buffer inputBuffer(opencl.queue, std::begin(input), std::end(input), false);
    opencl.queue.finish();

    auto t2 = clock_type::now();
    int localSize = 256;

    std::ofstream out("out.txt");

    cl::Kernel map(opencl.program, "map");
    cl::Buffer mapBuffer(opencl.context, CL_MEM_READ_WRITE, sizeof(float) * input.size());

    map.setArg(0, inputBuffer);
    map.setArg(1, mapBuffer);
    opencl.queue.enqueueNDRangeKernel(map, cl::NullRange, cl::NDRange(n), cl::NDRange(localSize));
    opencl.queue.finish();

    cl::copy(opencl.queue, mapBuffer, std::begin(indices), std::end(indices));

    out << "Map result: " << std::endl;
    for (int i = 0; i < indices.size(); i++)
    {
        out << indices[i] << "(" << input[i] << ")"
            << " ";
    }

    out << std::endl;

    cl::Kernel scanPartial(opencl.program, "scan_partial");
    cl::Buffer scanBuffer(opencl.context, CL_MEM_READ_WRITE, sizeof(float) * input.size());

    scanPartial.setArg(0, mapBuffer);
    scanPartial.setArg(1, scanBuffer);
    opencl.queue.enqueueNDRangeKernel(scanPartial, cl::NullRange, cl::NDRange(n), cl::NDRange(localSize));
    opencl.queue.finish();

    cl::copy(opencl.queue, scanBuffer, std::begin(indices), std::end(indices));

    out << "Scan partial result: " << std::endl;
    for (int i = 0; i < indices.size(); i++)
    {
        out << indices[i]
            << " ";
    }

    out << std::endl;

    cl::Kernel scanTotal(opencl.program, "scan_total");

    scanTotal.setArg(0, scanBuffer);
    opencl.queue.enqueueNDRangeKernel(scanTotal, cl::NullRange, cl::NDRange(n), cl::NDRange(localSize));
    opencl.queue.finish();

    cl::copy(opencl.queue, scanBuffer, std::begin(indices), std::end(indices));

    out << "Scan total result: " << std::endl;
    for (int i = 0; i < indices.size(); i++)
    {
        out << indices[i] << " ";
    }

    out << std::endl;

    cl::Kernel scatter(opencl.program, "scatter");
    cl::Buffer resultBuffer(opencl.context, CL_MEM_READ_WRITE, sizeof(float) * result.size());

    scatter.setArg(0, inputBuffer);
    scatter.setArg(1, scanBuffer);
    scatter.setArg(2, resultBuffer);

    opencl.queue.enqueueNDRangeKernel(scatter, cl::NullRange, cl::NDRange(n), cl::NDRange(localSize));
    opencl.queue.finish();
    auto t3 = clock_type::now();
    cl::copy(opencl.queue, resultBuffer, std::begin(result), std::end(result));
    auto t4 = clock_type::now();

    int actualSize = result[n];
    result.resize(actualSize);
    out << "Actual: ";
    for (auto i : result)
    {
        out << i << " ";
    }
    out << std::endl;
    verify_vector(expected_result, result);
    print("filter", {t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3});
}

void opencl_main(OpenCL &opencl)
{
    using namespace std::chrono;
    print_column_names();
    profile_filter(1024, opencl);
}

int main()
{
    try
    {
        // find OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
        {
            std::cerr << "Unable to find OpenCL platforms\n";
            return 1;
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
        try
        {
            program.build(devices);
        }
        catch (const cl::Error &err)
        {
            for (const auto &device : devices)
            {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }
        cl::CommandQueue queue(context, device);
        OpenCL opencl{platform, device, context, program, queue};
        opencl_main(opencl);
    }
    catch (const cl::Error &err)
    {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
                  << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return 1;
    }
    catch (const std::exception &err)
    {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}
