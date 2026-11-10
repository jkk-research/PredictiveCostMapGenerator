#pragma once

#include "plan_predictive_cost_map/casadi_function.hpp"
#include <nlopt.hpp>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>

namespace pcm
{
struct SolverResult
{
  Eigen::VectorXd coefficients;
  Eigen::VectorXd constraints;
  double objective = std::numeric_limits<double>::quiet_NaN();
  bool success = false;
  std::string status;
  int iterations = 0;
  int value_calls = 0;
  int derivative_calls = 0;
};

class NativeSLSQP
{
public:
  NativeSLSQP(const Function & values, const Function & derivatives, double tolerance,
              int maximum_evaluations = 120)
  : values_(values), derivatives_(derivatives), tolerance_(tolerance), maximum_evaluations_(maximum_evaluations) {}

  SolverResult solve(const Eigen::VectorXd & initial, const Eigen::VectorXd & parameters)
  {
    Context context{this, parameters};
    std::vector<double> coefficients(initial.data(), initial.data() + initial.size());
    evaluate(context, coefficients, false);
    nlopt::opt optimizer(nlopt::LD_SLSQP, initial.size());
    optimizer.set_min_objective(objective_callback, &context);
    optimizer.add_inequality_mconstraint(constraint_callback, &context,
                                         std::vector<double>(context.constraints.size(), tolerance_));
    optimizer.set_ftol_abs(tolerance_);
    optimizer.set_maxeval(maximum_evaluations_);
    nlopt::result status = nlopt::FAILURE;
    double objective = context.objective;
    try {status = optimizer.optimize(coefficients, objective);}
    catch (const nlopt::roundoff_limited &) {status = nlopt::ROUNDOFF_LIMITED;}
    catch (const std::exception & error) {
      return {initial, context.constraints, context.objective, false, error.what(),
              context.derivative_calls, context.value_calls, context.derivative_calls};
    }
    evaluate(context, coefficients, false);
    const bool converged = status == nlopt::SUCCESS || status == nlopt::STOPVAL_REACHED ||
                           status == nlopt::FTOL_REACHED || status == nlopt::XTOL_REACHED;
    Eigen::VectorXd result = Eigen::Map<Eigen::VectorXd>(coefficients.data(), coefficients.size());
    const bool valid = converged && result.allFinite() && std::isfinite(context.objective) &&
      context.constraints.allFinite() && context.constraints.maxCoeff() <= std::max(1e-6, 2.0 * tolerance_);
    return {result, context.constraints, context.objective, valid,
            valid ? "Solve_Succeeded" : "native_slsqp_status_" + std::to_string(static_cast<int>(status)),
            context.derivative_calls, context.value_calls, context.derivative_calls};
  }

private:
  struct Context
  {
    NativeSLSQP * solver;
    Eigen::VectorXd parameters;
    std::vector<double> point;
    double objective = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd constraints, gradient;
    Eigen::MatrixXd jacobian;
    bool has_values = false, has_derivatives = false;
    int value_calls = 0, derivative_calls = 0;
  };

  static void evaluate(Context & context, const std::vector<double> & point, bool derivatives)
  {
    if (context.point != point) {
      context.point = point; context.has_values = false; context.has_derivatives = false;
    }
    Eigen::VectorXd controls = Eigen::Map<const Eigen::VectorXd>(point.data(), point.size());
    if (derivatives && !context.has_derivatives) {
      auto result = context.solver->derivatives_({controls, context.parameters});
      context.objective = result[0](0, 0); context.gradient = result[1];
      context.constraints = result[2]; context.jacobian = result[3];
      context.has_values = true; context.has_derivatives = true; ++context.derivative_calls;
    } else if (!context.has_values) {
      auto result = context.solver->values_({controls, context.parameters});
      context.objective = result[0](0, 0); context.constraints = result[1];
      context.has_values = true; ++context.value_calls;
    }
    if (!std::isfinite(context.objective) || !context.constraints.allFinite() ||
        (derivatives && (!context.gradient.allFinite() || !context.jacobian.allFinite()))) {
      throw std::runtime_error("Native solver encountered nonfinite model outputs");
    }
  }

  static double objective_callback(const std::vector<double> & point, std::vector<double> & gradient, void * data)
  {
    auto & context = *static_cast<Context *>(data);
    evaluate(context, point, !gradient.empty());
    if (!gradient.empty()) {std::copy(context.gradient.data(), context.gradient.data() + context.gradient.size(), gradient.begin());}
    return context.objective;
  }

  static void constraint_callback(unsigned count, double * residual, unsigned dimension,
                                  const double * point, double * gradient, void * data)
  {
    auto & context = *static_cast<Context *>(data);
    evaluate(context, std::vector<double>(point, point + dimension), gradient != nullptr);
    if (count != static_cast<unsigned>(context.constraints.size())) {throw std::runtime_error("Native constraint size changed");}
    for (unsigned row = 0; row < count; ++row) {
      residual[row] = context.constraints[row];
      if (gradient) {
        for (unsigned column = 0; column < dimension; ++column) {gradient[row * dimension + column] = context.jacobian(row, column);}
      }
    }
  }

  Function values_, derivatives_;
  double tolerance_;
  int maximum_evaluations_;
};
}