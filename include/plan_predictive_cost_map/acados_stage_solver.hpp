#pragma once

#include <Eigen/Dense>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcm
{
class AcadosStages
{
public:
  struct Result
  {
    Eigen::VectorXd commands;
    Eigen::MatrixXd states;
    int status = -1, iterations = 0;
  };

  AcadosStages(const std::string & library, const std::string & dependencies, int count) : count_(count)
  {
    for (const auto & name : {"libblasfeo.so", "libhpipm.so", "libacados.so"}) {
      void * handle = dlopen((dependencies + "/" + name).c_str(), RTLD_NOW | RTLD_GLOBAL);
      if (!handle) {throw std::runtime_error(dlerror());}
      libraries_.push_back(handle);
    }
    void * handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {throw std::runtime_error(dlerror());}
    libraries_.push_back(handle);
    create_ = reinterpret_cast<void * (*)()>(dlsym(handle, "pcm_stage_create"));
    free_ = reinterpret_cast<void (*)(void *)>(dlsym(handle, "pcm_stage_free"));
    solve_ = reinterpret_cast<int (*)(void *, const double *, const double *, double *, double *, int *)>(dlsym(handle, "pcm_stage_solve"));
    if (!create_ || !free_ || !solve_) {throw std::runtime_error("Missing native acados stage API");}
    capsule_ = create_();
    if (!capsule_) {throw std::runtime_error("Native acados stage initialization failed");}
  }

  ~AcadosStages()
  {
    if (capsule_) {free_(capsule_);}
    for (auto iterator = libraries_.rbegin(); iterator != libraries_.rend(); ++iterator) {dlclose(*iterator);}
  }
  AcadosStages(const AcadosStages &) = delete;
  AcadosStages & operator=(const AcadosStages &) = delete;

  Result solve(const Eigen::VectorXd & parameters, const Eigen::VectorXd & initial,
                const Eigen::VectorXd & commands, const Eigen::MatrixXd & states)
  {
    if (initial.size() != 15 || commands.size() != count_ || states.rows() != 15 || states.cols() != count_ + 1) {
      throw std::runtime_error("Native acados stage dimensions do not match generated model");
    }
    Result result{commands, states};
    result.status = solve_(capsule_, parameters.data(), initial.data(), result.commands.data(), result.states.data(), &result.iterations);
    return result;
  }

private:
  int count_;
  std::vector<void *> libraries_;
  void * capsule_ = nullptr;
  void * (*create_)() = nullptr;
  void (*free_)(void *) = nullptr;
  int (*solve_)(void *, const double *, const double *, double *, double *, int *) = nullptr;
};
}