#include "plan_predictive_cost_map/native_solver.hpp"
#include <iomanip>
#include <iostream>

int main(int argc, char ** argv)
{
  pcm::NativeSLSQP solver(pcm::Function::load(std::string(argv[1]) + "/fg.casadi"),
                           pcm::Function::load(std::string(argv[1]) + "/jac.casadi"), 1e-8);
  Eigen::Vector2d initial(0.0, 0.0), parameters(1.0, 1.0);
  auto result = solver.solve(initial, parameters);
  if (!result.success) {std::cerr << result.status << '\n'; return 1;}
  auto ipopt = pcm::Function::load(std::string(argv[1]) + "/ipopt.casadi").solve(initial, parameters);
  if (!ipopt.success || (ipopt.coefficients - result.coefficients).norm() > 1e-6) {return 2;}
  pcm::NativeSLSQP limited(pcm::Function::load(std::string(argv[1]) + "/fg.casadi"),
                          pcm::Function::load(std::string(argv[1]) + "/jac.casadi"), 1e-8, 1);
  if (limited.solve(initial, parameters).success) {return 3;}
  std::cout << std::setprecision(17) << result.coefficients.transpose() << ' ' << result.objective << '\n';
}