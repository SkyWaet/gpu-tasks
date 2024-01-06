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
#include <random>
#include <sstream>
#include <string>
#include <fstream>

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

void print(const char *name, std::array<duration, 5> dt, std::array<double, 2> bw)
{
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i = 0; i < 5; ++i)
    {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
        std::cout << std::setw(20) << tmp.str();
    }
    for (size_t i = 0; i < 2; ++i)
    {
        std::stringstream tmp;
        tmp << bw[i] << "GB/s";
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
    std::cout << std::setw(20) << "OpenMP bandwidth";
    std::cout << std::setw(20) << "OpenCL bandwidth";
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

void assertCloseTo(float expected, float actual)
{
    if (std::abs(expected - actual) > 1e3)
    {
        std::stringstream msg;
        msg << "Expected result to be close to: " << expected << ", but was: " << actual << ". Difference is " << std::abs(expected - actual);
        throw std::runtime_error(msg.str());
    }
}

void profile_reduce(int n, OpenCL &opencl)
{
    auto a = random_vector<float>(n);
    auto t0 = clock_type::now();
    float expected_result = reduce(a);
    auto t1 = clock_type::now();

    opencl.queue.flush();
    cl::Kernel kernel(opencl.program, "reduce");
    cl::Buffer vectorBuffer(opencl.queue, std::begin(a), std::end(a), true);
    opencl.queue.flush();

    auto t2 = clock_type::now();
    auto t3 = t2;
    auto t4 = t2;

    int localSize = 256;
    Vector<float> result(localSize);

    int resultSize = n / localSize;
    while(true)
    {
        kernel.setArg(0, vectorBuffer);
        cl::Buffer resultBuffer(opencl.context, CL_MEM_READ_WRITE, sizeof(float) * resultSize);
        kernel.setArg(1, resultBuffer);
        opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(resultSize * localSize), cl::NDRange(localSize));

        vectorBuffer = resultBuffer;

        if (resultSize % localSize != 0 && resultSize > localSize)
        {
            resultSize *= 2;
        }

        if (resultSize <= localSize)
        {
            opencl.queue.flush();
            t3 = clock_type::now();
            cl::copy(opencl.queue, resultBuffer, std::begin(result), std::begin(result) + resultSize);
            t4 = clock_type::now();
            break;
        }

        resultSize /= localSize;

    }

    assertCloseTo(expected_result, result[0]);

    print("reduce",
          {t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3},
          {bandwidth(n * n + n + n, t0, t1), bandwidth(n * n + n + n, t2, t3)});
}

void profile_scan_inclusive(int n)
{
    auto a = random_vector<float>(n);
    Vector<float> result(a), expected_result(a);
    auto t0 = clock_type::now();
    scan_inclusive(expected_result);
    auto t1 = clock_type::now();
    auto t2 = clock_type::now();
    auto t3 = clock_type::now();
    auto t4 = clock_type::now();
    // TODO Implement OpenCL version! See profile_vector_times_vector for an example.
    // TODO Uncomment the following line!
    // verify_vector(expected_result, result);
    print("scan-inclusive",
          {t1 - t0, t4 - t1, t2 - t1, t3 - t2, t4 - t3},
          {bandwidth(n * n + n * n + n * n, t0, t1), bandwidth(n * n + n * n + n * n, t2, t3)});
}

void opencl_main(OpenCL &opencl)
{
    using namespace std::chrono;
    print_column_names();
    profile_reduce(1024 * 1024 * 10, opencl);
    profile_scan_inclusive(1024 * 1024 * 10);
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
