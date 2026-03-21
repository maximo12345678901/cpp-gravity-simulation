#pragma once
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

struct float4 {
    float x, y, z, w;
    float4() : x(0), y(0), z(0), w(0) {}
    float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
};

class GravityGPU {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_device_id device;

    cl_mem bufPos, bufVel, bufNewVel, bufNewPos;
    int allocatedN = 0;
    bool valid = false;

    std::string loadKernel(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Could not open kernel file: " + path);
        }
        std::string src(std::istreambuf_iterator<char>(f), {});
        if (src.empty()) {
            throw std::runtime_error("Kernel file is empty: " + path);
        }
        std::cout << "Loaded kernel (" << src.size() << " bytes) from " << path << "\n";
        return src;
    }

    void allocBuffers(int n) {
        if (n == allocatedN) return;
        if (allocatedN > 0) {
            clReleaseMemObject(bufPos);    clReleaseMemObject(bufVel);
            clReleaseMemObject(bufNewVel); clReleaseMemObject(bufNewPos);
        }
        cl_int err;
        bufPos    = clCreateBuffer(context, CL_MEM_READ_ONLY,  n * sizeof(float4), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateBuffer bufPos failed: " + std::to_string(err));
        bufVel    = clCreateBuffer(context, CL_MEM_READ_ONLY,  n * sizeof(float4), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateBuffer bufVel failed: " + std::to_string(err));
        bufNewVel = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float4), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateBuffer bufNewVel failed: " + std::to_string(err));
        bufNewPos = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float4), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("clCreateBuffer bufNewPos failed: " + std::to_string(err));
        allocatedN = n;
    }

public:
    GravityGPU() {
        cl_int err;

        cl_platform_id platform;
        cl_uint numPlatforms;
        err = clGetPlatformIDs(1, &platform, &numPlatforms);
        if (err != CL_SUCCESS || numPlatforms == 0)
            throw std::runtime_error("No OpenCL platforms found. Error: " + std::to_string(err));
        std::cout << "Found " << numPlatforms << " OpenCL platform(s)\n";

        cl_uint numDevices;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
        if (err != CL_SUCCESS || numDevices == 0)
            throw std::runtime_error("No GPU devices found. Error: " + std::to_string(err));

        char deviceName[256];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
        std::cout << "Using GPU: " << deviceName << "\n";

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("clCreateContext failed: " + std::to_string(err));

        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("clCreateCommandQueue failed: " + std::to_string(err));

        std::string src = loadKernel("gravity.cl");
        const char* srcPtr = src.c_str();
        size_t srcLen = src.size();
        program = clCreateProgramWithSource(context, 1, &srcPtr, &srcLen, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("clCreateProgramWithSource failed: " + std::to_string(err));

        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char log[8192];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
            throw std::runtime_error(std::string("Kernel build failed:\n") + log);
        }
        std::cout << "Kernel compiled successfully\n";

        kernel = clCreateKernel(program, "computeForces", &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("clCreateKernel failed: " + std::to_string(err));

        valid = true;
    }

    void step(std::vector<GravityObject>& objects, float dt, float G, float epsilon) {
        if (!valid) return;

        int n = objects.size();
        if (n == 0) return;

        allocBuffers(n);

        // Pack data
        std::vector<float4> pos(n), vel(n);
        for (int i = 0; i < n; ++i) {
            pos[i] = float4(objects[i].position.x, objects[i].position.y, objects[i].mass, objects[i].radius);
            vel[i] = float4(objects[i].velocity.x, objects[i].velocity.y, 0.f, 0.f);
        }

        // Upload
        clEnqueueWriteBuffer(queue, bufPos, CL_TRUE, 0, n*sizeof(float4), pos.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, bufVel, CL_TRUE, 0, n*sizeof(float4), vel.data(), 0, nullptr, nullptr);

        // Set args
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufPos);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufVel);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufNewVel);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufNewPos);
        clSetKernelArg(kernel, 4, sizeof(int),    &n);
        clSetKernelArg(kernel, 5, sizeof(float),  &dt);
        clSetKernelArg(kernel, 6, sizeof(float),  &G);
        clSetKernelArg(kernel, 7, sizeof(float),  &epsilon);

        // Run — make sure globalSize is a multiple of localSize
        size_t localSize  = 64;
        size_t globalSize = ((n + localSize - 1) / localSize) * localSize;
        cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "clEnqueueNDRangeKernel failed: " << err << "\n";
            return;
        }

        // Download
        std::vector<float4> newVel(n), newPos(n);
        clEnqueueReadBuffer(queue, bufNewVel, CL_TRUE, 0, n*sizeof(float4), newVel.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, bufNewPos, CL_TRUE, 0, n*sizeof(float4), newPos.data(), 0, nullptr, nullptr);

        // Unpack — skip nan results
        for (int i = 0; i < n; ++i) {
            if (!objects[i].isGrabbed) {
                if (std::isfinite(newPos[i].x) && std::isfinite(newPos[i].y) &&
                    std::isfinite(newVel[i].x) && std::isfinite(newVel[i].y)) {
                    objects[i].velocity = Vector2(newVel[i].x, newVel[i].y);
                    objects[i].position = Vector2(newPos[i].x, newPos[i].y);
                } else {
                    std::cerr << "NaN detected for object " << i << ", skipping update\n";
                }
            }
        }
    }

    ~GravityGPU() {
        if (allocatedN > 0) {
            clReleaseMemObject(bufPos);    clReleaseMemObject(bufVel);
            clReleaseMemObject(bufNewVel); clReleaseMemObject(bufNewPos);
        }
        if (valid) {
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }
    }
};
