#include <casadi/casadi.hpp>
#include <cstring>
#include <limits>

static thread_local char last_error[2048] = {};

extern "C" {
void * pcm_function_load(const char * filename)
{
  try {return new casadi::Function(casadi::Function::load(filename));}
  catch (const std::exception & error) {std::strncpy(last_error, error.what(), sizeof(last_error) - 1); return nullptr;}
}
void pcm_function_free(void * function) {delete static_cast<casadi::Function *>(function);}
const char * pcm_function_error() {return last_error;}
int pcm_function_outputs(void * function) {return static_cast<casadi::Function *>(function)->n_out();}
int pcm_function_rows(void * function, int output) {return static_cast<casadi::Function *>(function)->size1_out(output);}
int pcm_function_columns(void * function, int output) {return static_cast<casadi::Function *>(function)->size2_out(output);}
int pcm_function_evaluate(void * function, int count, const double ** input,
                          const int * rows, const int * columns, double ** output)
{
  try {
    std::vector<casadi::DM> arguments;
    for (int index = 0; index < count; ++index) {
      arguments.push_back(casadi::DM::reshape(casadi::DM(std::vector<double>(input[index], input[index] + rows[index] * columns[index])), rows[index], columns[index]));
    }
    auto result = (*static_cast<casadi::Function *>(function))(arguments);
    for (std::size_t index = 0; index < result.size(); ++index) {
      auto values = result[index].get_elements();
      std::copy(values.begin(), values.end(), output[index]);
    }
    return 0;
  } catch (const std::exception & error) {std::strncpy(last_error, error.what(), sizeof(last_error) - 1); return 1;}
}
int pcm_solver_evaluate(void * function, const double * initial, int decisions,
                         const double * parameters, int parameter_count, double * solution,
                         double * objective, double * constraints, int constraint_count,
                         int * success, int * iterations, char * status, int status_size)
{
  try {
    auto & solver = *static_cast<casadi::Function *>(function);
    casadi::DMDict arguments{{"x0", casadi::DM(std::vector<double>(initial, initial + decisions))},
      {"p", casadi::DM(std::vector<double>(parameters, parameters + parameter_count))},
      {"lbx", -std::numeric_limits<double>::infinity()}, {"ubx", std::numeric_limits<double>::infinity()},
      {"lbg", -std::numeric_limits<double>::infinity()}, {"ubg", 0.0}};
    auto result = solver(arguments);
    auto coefficients = result.at("x").get_elements(), residuals = result.at("g").get_elements();
    if (coefficients.size() != static_cast<std::size_t>(decisions) || residuals.size() != static_cast<std::size_t>(constraint_count)) {
      throw std::runtime_error("Native solver output dimensions changed");
    }
    std::copy(coefficients.begin(), coefficients.end(), solution);
    std::copy(residuals.begin(), residuals.end(), constraints);
    *objective = result.at("f").scalar();
    auto stats = solver.stats();
    *success = stats.at("success").to_bool(); *iterations = stats.at("iter_count").to_int();
    std::strncpy(status, stats.at("return_status").to_string().c_str(), status_size - 1);
    return 0;
  } catch (const std::exception & error) {std::strncpy(last_error, error.what(), sizeof(last_error) - 1); return 1;}
}
}