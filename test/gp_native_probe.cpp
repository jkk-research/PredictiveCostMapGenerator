#include "plan_predictive_cost_map/gp.hpp"
#include <iomanip>
#include <iostream>

int main()
{
  Eigen::MatrixXd points(10, 3);
  for (int index = 0; index < 10; ++index) {points.row(index) << 3.0 * index, -2.0 + 0.2 * index, 0.5 * index;}
  Eigen::VectorXd parameters(4);
  parameters << 0.8, 2.7, 2.2, 1.0;
  auto matrix = pcm::kernel(points, points, parameters);
  Eigen::VectorXd targets = Eigen::VectorXd::LinSpaced(10, 1.0, 1.8);
  Eigen::VectorXd noise = Eigen::VectorXd::LinSpaced(10, 0.05, 0.2);
  auto alpha = pcm::fit(matrix, targets, noise);
  auto projection = pcm::compress(points, alpha, parameters, {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}, 5, matrix);
  std::cout << std::setprecision(17);
  for (int index : projection.selected) {std::cout << index << ' ';}
  for (double value : alpha) {std::cout << value << ' ';}
  for (double value : projection.coefficients) {std::cout << value << ' ';}
  std::cout << projection.error_bound << '\n';
}