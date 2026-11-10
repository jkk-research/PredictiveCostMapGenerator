#pragma once

#include "plan_predictive_cost_map/gp.hpp"
#include "plan_predictive_cost_map/casadi_function.hpp"
#include <crp_msgs/msg/scenario.hpp>

namespace pcm
{
inline Eigen::MatrixXd eigen(const Eigen::MatrixXd & value) {return value;}

inline Eigen::MatrixXd dm(const Eigen::MatrixXd & values) {return values;}

inline double yaw(const geometry_msgs::msg::Quaternion & quaternion)
{
  return std::atan2(2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
                   1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z));
}

inline double interpolate(const std::vector<double> & times, const std::vector<double> & values, double time)
{
  if (time <= times.front()) {return values.front();}
  if (time >= times.back()) {return values.back();}
  const auto upper = std::upper_bound(times.begin(), times.end(), time) - times.begin();
  const double fraction = (time - times[upper - 1]) / (times[upper] - times[upper - 1]);
  return values[upper - 1] + fraction * (values[upper] - values[upper - 1]);
}

class Reference
{
public:
  Reference(const Function & basis, int count) : count_(count)
  {
    Eigen::MatrixXd matrix(count, count);
    for (int index = 0; index < count; ++index) {
      matrix.row(index) = basis(std::vector<Eigen::MatrixXd>{scalar(double(index) / (count - 1))})[0].transpose();
    }
    inverse_ = matrix.inverse();
  }

  Eigen::VectorXd prepare(const Eigen::MatrixXd & input) const
  {
    std::vector<Eigen::Vector3d> points;
    for (int index = 0; index < input.rows(); ++index) {
      Eigen::Vector3d point = input.row(index);
      if (points.empty() || (point - points.back()).norm() > 1e-8) {points.push_back(point);}
    }
    if (points.size() < 2) {throw std::runtime_error("Target reference has insufficient geometry");}
    std::vector<double> distance{0.0};
    std::vector<std::vector<double>> coordinates(3);
    double nearest = std::numeric_limits<double>::infinity(), origin = 0.0;
    for (std::size_t index = 0; index < points.size(); ++index) {
      for (int axis = 0; axis < 3; ++axis) {coordinates[axis].push_back(points[index][axis]);}
      if (index) {
        Eigen::Vector3d delta = points[index] - points[index - 1];
        const double length = delta.norm();
        double fraction = std::clamp(-points[index - 1].dot(delta) / delta.squaredNorm(), 0.0, 1.0);
        double norm = (points[index - 1] + fraction * delta).squaredNorm();
        if (norm < nearest) {nearest = norm; origin = distance.back() + fraction * length;}
        distance.push_back(distance.back() + length);
      }
    }
    const double step = distance.back() / (count_ - 1);
    Eigen::MatrixXd values(count_, 6);
    for (int index = 0; index < count_; ++index) {
      for (int axis = 0; axis < 3; ++axis) {
        values(index, axis) = interpolate(distance, coordinates[axis], index * step);
      }
    }
    auto derivative = [&](const Eigen::VectorXd & column, int index) {
        if (!index) {return (column[1] - column[0]) / step;}
        if (index == count_ - 1) {return (column[index] - column[index - 1]) / step;}
        return (column[index + 1] - column[index - 1]) / (2.0 * step);
      };
    for (int index = 0; index < count_; ++index) {
      double tangent_x = derivative(values.col(0), index), tangent_y = derivative(values.col(1), index);
      double heading = std::atan2(tangent_y, tangent_x);
      if (index) {heading = values(index - 1, 3) + std::remainder(heading - values(index - 1, 3), 2 * M_PI);}
      values(index, 3) = heading;
      values(index, 4) = std::atan2(derivative(values.col(2), index), std::max(std::hypot(tangent_x, tangent_y), 1e-8));
    }
    for (int index = 0; index < count_; ++index) {values(index, 5) = derivative(values.col(3), index);}
    Eigen::MatrixXd coefficients = (inverse_ * values).transpose();
    Eigen::VectorXd result(2 + 6 * count_);
    result[0] = -origin; result[1] = distance.back() - origin;
    result.tail(6 * count_) = Eigen::Map<Eigen::VectorXd>(coefficients.data(), coefficients.size());
    return result;
  }

private:
  int count_;
  Eigen::MatrixXd inverse_;
};

struct GP
{
  Eigen::MatrixXd points;
  Eigen::VectorXd alpha;
  double error = 0.0;
};

inline GP static_gp(const Eigen::MatrixXd & input, int count, const Eigen::VectorXd & parameters,
                    double severity, double noise)
{
  std::vector<Eigen::Vector2d> unique;
  for (int index = 0; index < input.rows(); ++index) {unique.push_back(input.row(index).head(2));}
  std::sort(unique.begin(), unique.end(), [](const auto & left, const auto & right) {
      return left[0] < right[0] || (left[0] == right[0] && left[1] < right[1]);});
  unique.erase(std::unique(unique.begin(), unique.end(), [](const auto & left, const auto & right) {
      return left == right;}), unique.end());
  GP result{Eigen::MatrixXd::Constant(count, 2, 1000.0), Eigen::VectorXd::Zero(count)};
  if (unique.empty()) {return result;}
  std::vector<int> selected;
  if (static_cast<int>(unique.size()) > count) {
    std::vector<double> distance(unique.size());
    int nearest = 0;
    for (std::size_t index = 1; index < unique.size(); ++index) {
      if (unique[index].squaredNorm() < unique[nearest].squaredNorm()) {nearest = index;}
    }
    selected.push_back(nearest);
    for (std::size_t index = 0; index < unique.size(); ++index) {distance[index] = (unique[index] - unique[nearest]).squaredNorm();}
    while (static_cast<int>(selected.size()) < count) {
      int index = std::max_element(distance.begin(), distance.end()) - distance.begin();
      selected.push_back(index);
      for (std::size_t candidate = 0; candidate < unique.size(); ++candidate) {
        distance[candidate] = std::min(distance[candidate], (unique[candidate] - unique[index]).squaredNorm());
      }
    }
  } else {
    for (std::size_t index = 0; index < unique.size(); ++index) {selected.push_back(index);}
  }
  Eigen::VectorXd targets = Eigen::VectorXd::Zero(count);
  for (std::size_t index = 0; index < selected.size(); ++index) {result.points.row(index) = unique[selected[index]]; targets[index] = severity;}
  result.alpha = fit(kernel(result.points, result.points, parameters), targets, Eigen::VectorXd::Constant(count, noise * noise));
  return result;
}
}