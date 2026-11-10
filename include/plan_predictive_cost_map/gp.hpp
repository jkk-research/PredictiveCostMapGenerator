#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pcm
{
inline Eigen::MatrixXd kernel(const Eigen::MatrixXd & query, const Eigen::MatrixXd & support,
                             const Eigen::VectorXd & parameters)
{
  Eigen::MatrixXd result(query.rows(), support.rows());
  for (Eigen::Index row = 0; row < query.rows(); ++row) {
    for (Eigen::Index column = 0; column < support.rows(); ++column) {
      const double distance = ((query.row(row) - support.row(column)).array() /
        parameters.tail(query.cols()).transpose().array()).square().sum();
      result(row, column) = parameters[0] * parameters[0] * std::exp(-0.5 * distance);
    }
  }
  return result;
}

inline Eigen::VectorXd fit(const Eigen::MatrixXd & matrix, const Eigen::VectorXd & targets,
                           const Eigen::VectorXd & noise)
{
  Eigen::MatrixXd regularized = matrix;
  regularized.diagonal() += noise;
  Eigen::VectorXd alpha = regularized.ldlt().solve(targets);
  if (!alpha.allFinite() || (regularized * alpha - targets).norm() > 1e-6 * (1.0 + targets.norm())) {
    throw std::runtime_error("GP fit failed");
  }
  return alpha;
}

struct Projection
{
  std::vector<int> selected;
  Eigen::VectorXd coefficients;
  double error_bound = 0.0;
};

inline Projection compress(const Eigen::MatrixXd & points, const Eigen::VectorXd & alpha,
                           const Eigen::VectorXd & parameters, const std::vector<std::vector<int>> & groups,
                           int supports_per_path, const Eigen::MatrixXd & matrix)
{
  Projection result;
  Eigen::MatrixXd scaled = points.array().rowwise() / parameters.tail(points.cols()).transpose().array();
  bool exact = true;
  for (const auto & group : groups) {
    if (group.empty()) {continue;}
    exact = exact && supports_per_path >= static_cast<int>(group.size());
    std::vector<int> chosen{0};
    Eigen::VectorXd distance(group.size());
    auto squared = [&](int left, int right) {
        return (scaled.row(group[left]) - scaled.row(group[right])).squaredNorm();
      };
    for (std::size_t index = 0; index < group.size(); ++index) {distance[index] = squared(index, 0);}
    for (int iteration = 1; iteration < std::min(supports_per_path, static_cast<int>(group.size())); ++iteration) {
      for (int index : chosen) {distance[index] = -1.0;}
      Eigen::Index index;
      distance.maxCoeff(&index);
      chosen.push_back(static_cast<int>(index));
      for (std::size_t candidate = 0; candidate < group.size(); ++candidate) {
        distance[candidate] = std::min(distance[candidate], squared(candidate, index));
      }
    }
    std::sort(chosen.begin(), chosen.end());
    for (int index : chosen) {result.selected.push_back(group[index]);}
  }
  result.coefficients.resize(result.selected.size());
  if (result.selected.empty()) {return result;}
  if (exact) {
    for (std::size_t index = 0; index < result.selected.size(); ++index) {
      result.coefficients[index] = alpha[result.selected[index]];
    }
    return result;
  }
  Eigen::MatrixXd reduced(result.selected.size(), result.selected.size());
  Eigen::VectorXd cross(result.selected.size());
  for (std::size_t row = 0; row < result.selected.size(); ++row) {
    cross[row] = matrix.row(result.selected[row]).dot(alpha);
    for (std::size_t column = 0; column < result.selected.size(); ++column) {
      reduced(row, column) = matrix(result.selected[row], result.selected[column]);
    }
  }
  result.coefficients = fit(reduced, cross, Eigen::VectorXd::Constant(reduced.rows(),
                           parameters[0] * parameters[0] * 1e-10));
  Eigen::VectorXd residual = alpha;
  for (std::size_t index = 0; index < result.selected.size(); ++index) {
    residual[result.selected[index]] -= result.coefficients[index];
  }
  const double energy = residual.dot(matrix * residual);
  const double roundoff = 64.0 * alpha.size() * std::numeric_limits<double>::epsilon() *
    residual.cwiseAbs().dot(matrix.cwiseAbs() * residual.cwiseAbs());
  result.error_bound = parameters[0] * std::sqrt(std::max(0.0, energy) + roundoff);
  if (!std::isfinite(result.error_bound)) {throw std::runtime_error("Invalid GP projection bound");}
  return result;
}
}