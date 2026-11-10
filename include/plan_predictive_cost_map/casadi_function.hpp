#pragma once
#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <vector>

extern "C" {
void * pcm_function_load(const char * filename);
void pcm_function_free(void * function);
const char * pcm_function_error();
int pcm_function_outputs(void * function);
int pcm_function_rows(void * function, int output);
int pcm_function_columns(void * function, int output);
int pcm_function_evaluate(void * function, int count, const double ** input,
                          const int * rows, const int * columns, double ** output);
int pcm_solver_evaluate(void * function, const double * initial, int decisions,
                         const double * parameters, int parameter_count, double * solution,
                         double * objective, double * constraints, int constraint_count,
                         int * success, int * iterations, char * status, int status_size);
}

namespace pcm
{
inline Eigen::MatrixXd scalar(double value) {return Eigen::MatrixXd::Constant(1, 1, value);}

class Function
{
public:
  static Function load(const std::string & filename)
  {
    Function result;
    result.handle_ = std::shared_ptr<void>(pcm_function_load(filename.c_str()), pcm_function_free);
    if (!result.handle_) {throw std::runtime_error(pcm_function_error());}
    return result;
  }

  struct Solution
  {
    Eigen::VectorXd coefficients, constraints;
    double objective = 0.0;
    bool success = false;
    int iterations = 0;
    std::string status;
  };

  Solution solve(const Eigen::VectorXd & initial, const Eigen::VectorXd & parameters) const
  {
    Solution result;
    result.coefficients.resize(initial.size());
    result.constraints.resize(pcm_function_rows(handle_.get(), 2) * pcm_function_columns(handle_.get(), 2));
    int success = 0;
    char status[256] = {};
    if (pcm_solver_evaluate(handle_.get(), initial.data(), initial.size(), parameters.data(), parameters.size(),
        result.coefficients.data(), &result.objective, result.constraints.data(), result.constraints.size(),
        &success, &result.iterations, status, sizeof(status))) {throw std::runtime_error(pcm_function_error());}
    result.success = success != 0; result.status = status;
    return result;
  }

  std::vector<Eigen::MatrixXd> operator()(const std::vector<Eigen::MatrixXd> & arguments) const
  {
    std::vector<const double *> inputs;
    std::vector<int> input_rows, input_columns;
    for (const auto & argument : arguments) {
      inputs.push_back(argument.data()); input_rows.push_back(argument.rows()); input_columns.push_back(argument.cols());
    }
    std::vector<Eigen::MatrixXd> result;
    std::vector<double *> output;
    for (int index = 0; index < pcm_function_outputs(handle_.get()); ++index) {
      result.emplace_back(pcm_function_rows(handle_.get(), index), pcm_function_columns(handle_.get(), index));
    }
    for (auto & values : result) {output.push_back(values.data());}
    if (pcm_function_evaluate(handle_.get(), inputs.size(), inputs.data(), input_rows.data(), input_columns.data(), output.data())) {
      throw std::runtime_error(pcm_function_error());
    }
    return result;
  }

private:
  std::shared_ptr<void> handle_;
};
}