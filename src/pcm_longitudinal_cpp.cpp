#include "plan_predictive_cost_map/native_preprocessing.hpp"
#include "plan_predictive_cost_map/native_solver.hpp"
#include "plan_predictive_cost_map/acados_stage_solver.hpp"
#include "native_json.hpp"
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <cstring>
#include <crp_msgs/msg/ego.hpp>
#include <crp_msgs/msg/actuator_state.hpp>
#include <autoware_control_msgs/msg/control.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <std_msgs/msg/string.hpp>
#include <fstream>
#include <deque>
#include <mutex>
#include <numeric>

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

static std::vector<double> flatten(const Eigen::MatrixXd & matrix)
{
  return {matrix.data(), matrix.data() + matrix.size()};
}

static std::vector<std::vector<double>> rows(const Eigen::MatrixXd & matrix)
{
  std::vector<std::vector<double>> result;
  for (int row = 0; row < matrix.rows(); ++row) {
    std::vector<double> values;
    for (int column = 0; column < matrix.cols(); ++column) {values.push_back(matrix(row, column));}
    result.push_back(values);
  }
  return result;
}

struct EgoSample
{
  int64_t stamp = 0;
  double speed = 0.0, lateral = 0.0, yaw_rate = 0.0, acceleration = 0.0, steering = 0.0;
  double x = 0.0, y = 0.0, z = 0.0, heading = 0.0, pitch = 0.0;
};

struct Scene
{
  crp_msgs::msg::Scenario::ConstSharedPtr message;
  Eigen::MatrixXd route, bounds;
  Eigen::VectorXd reference;
  pcm::GP target, road;
  EgoSample origin;
  bool origin_valid = false;
  double target_speed = 0.0;
};

class NativePlanner : public rclcpp::Node
{
public:
  NativePlanner(const rclcpp::NodeOptions & options = rclcpp::NodeOptions()) : Node("pcm_longitudinal_cpp", options)
  {
    const auto directory = declare_parameter<std::string>("model_directory", "");
    if (directory.empty()) {throw std::runtime_error("model_directory is required; use pcm_longitudinal_cpp.launch.py");}
    std::ifstream stream(directory + "/manifest.json");
    std::string encoded((std::istreambuf_iterator<char>(stream)), {});
    Json metadata = Json::parse(encoded);
    Json parameters = metadata["parameters"];
    rcl_interfaces::msg::ParameterDescriptor model_parameter;
    model_parameter.read_only = true;
    model_parameter.description = "Generated model setting; restart with regenerated functions to change";
    for (auto & item : parameters.items()) {
      std::string name = item.key();
      if (has_parameter(name)) {continue;}
      auto & value = item.value();
      if (value.is_boolean()) {declare_parameter(name, value.get<bool>(), model_parameter);}
      else if (value.is_number_integer()) {declare_parameter(name, value.get<int64_t>(), model_parameter);}
      else if (value.is_number_float()) {declare_parameter(name, value.get<double>(), model_parameter);}
      else if (value.is_string()) {declare_parameter(name, value.get<std::string>(), model_parameter);}
      else {declare_parameter(name, value.get<std::vector<double>>(), model_parameter);}
    }
    actuators_ = metadata["actuators_enabled"].get<bool>();
    exact_gp_capacity_ = metadata.value("exact_gp_capacity", actuators_);
    decisions_ = metadata["decision_count"].get<int>();
    parameter_count_ = metadata["parameter_count"].get<int>();
    prediction_outputs_ = metadata["prediction_output_count"].get<int>();
    pedestrian_count_ = metadata["pedestrian_support_count"].get<int>();
    vehicle_count_ = metadata["vehicle_support_count"].get<int>();
    horizon_ = real("lookahead_time"); steps_ = integer("prediction_steps"); period_ = real("timer_period");
    previous_ = Eigen::VectorXd::Zero(decisions_);
    staged_ = metadata.value("velocity_representation", std::string("polynomial")) == "acados_stages";
    if (staged_) {
      stage_ = pcm::Function::load(directory + "/stage.casadi");
      stage_solver_ = std::make_unique<pcm::AcadosStages>(metadata["acados_runtime_library"].get<std::string>(),
                                                        metadata["acados_lib_dir"].get<std::string>(), steps_);
    } else {
      fg_ = pcm::Function::load(directory + "/fg.casadi");
      reporting_ = pcm::Function::load(directory + "/report.casadi");
    }
    reference_ = std::make_unique<pcm::Reference>(pcm::Function::load(directory + "/basis.casadi"), integer("reference_samples"));
    native_sqp_ = metadata["solver_method"].get<std::string>() == "sqp";
    if (native_sqp_) {
      optimizer_ = std::make_unique<pcm::NativeSLSQP>(fg_, pcm::Function::load(directory + "/jac.casadi"), real("solver_tolerance"));
      retry_ = pcm::Function::load(directory + "/retry.casadi");
    } else if (!staged_) {
      primary_ = pcm::Function::load(directory + "/solver.casadi");
      warm_ = pcm::Function::load(directory + "/warm.casadi");
    }
    command_pub_ = create_publisher<autoware_control_msgs::msg::Control>("/control/command/control_cmd", 30);
    trajectory_pub_ = create_publisher<autoware_planning_msgs::msg::Trajectory>("/plan/trajectory", 30);
    timing_pub_ = create_publisher<std_msgs::msg::String>("/plan/pcm_longitudinal/timing", rclcpp::QoS(1).best_effort());
    gp_pub_ = create_publisher<std_msgs::msg::String>("/plan/pcm_longitudinal/gp_state", rclcpp::QoS(1).best_effort());
    telemetry_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    scene_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions telemetry_options, scene_options;
    telemetry_options.callback_group = telemetry_group_; scene_options.callback_group = scene_group_;
    ego_sub_ = create_subscription<crp_msgs::msg::Ego>("/ego", 1, [this](crp_msgs::msg::Ego::ConstSharedPtr message) {
      EgoSample sample;
      sample.stamp = rclcpp::Time(message->header.stamp).nanoseconds();
      sample.speed = message->twist.twist.linear.x; sample.lateral = message->twist.twist.linear.y;
      sample.yaw_rate = message->twist.twist.angular.z; sample.acceleration = message->accel.accel.linear.x;
      sample.steering = message->tire_angle_front; sample.x = message->pose.pose.position.x;
      sample.y = message->pose.pose.position.y; sample.z = message->pose.pose.position.z;
      const auto & quaternion = message->pose.pose.orientation;
      sample.heading = pcm::yaw(quaternion);
      sample.pitch = -std::asin(std::clamp(2 * (quaternion.w * quaternion.y - quaternion.z * quaternion.x), -1.0, 1.0));
      std::lock_guard<std::mutex> lock(mutex_); ego_ = sample; history_.push_back(sample);
      if (history_.size() > 400) {history_.pop_front();}
    }, telemetry_options);
    actuator_sub_ = create_subscription<crp_msgs::msg::ActuatorState>("/simulator/actuator_state", 1,
      [this](crp_msgs::msg::ActuatorState::ConstSharedPtr message) {
        Eigen::Vector4d state(message->motor_speed_rad_s, message->shaft_twist_rad,
                              message->motor_torque_nm, message->filtered_acceleration_mps2);
        if (!state.allFinite() || state[0] < 0) {return;}
        std::lock_guard<std::mutex> lock(mutex_); powertrain_ = state; actuator_time_ = Clock::now();
      }, telemetry_options);
    scenario_sub_ = create_subscription<crp_msgs::msg::Scenario>("/scenario", 1,
      [this](crp_msgs::msg::Scenario::ConstSharedPtr message) {receive_scene(message);}, scene_options);
    timer_ = create_wall_timer(std::chrono::duration<double>(period_), [this]() {plan();});
    RCLCPP_INFO(get_logger(), "Native ROS/GP planner ready; native solver, no embedded Python");
  }

  Json model_probe(const std::string & directory, const std::vector<double> & coefficients)
  {
    auto deserialize = [&](const std::string & filename, auto & message) {
        std::ifstream stream(directory + "/" + filename, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(stream)), {});
        rclcpp::SerializedMessage serialized(bytes.size());
        auto & buffer = serialized.get_rcl_serialized_message();
        std::memcpy(buffer.buffer, bytes.data(), bytes.size()); buffer.buffer_length = bytes.size();
        rclcpp::Serialization<std::decay_t<decltype(message)>> serialization;
        serialization.deserialize_message(&serialized, &message);
      };
    crp_msgs::msg::Ego measured;
    deserialize("ego.cdr", measured);
    ego_.stamp = rclcpp::Time(measured.header.stamp).nanoseconds();
    ego_.speed = measured.twist.twist.linear.x; ego_.lateral = measured.twist.twist.linear.y;
    ego_.yaw_rate = measured.twist.twist.angular.z; ego_.acceleration = measured.accel.accel.linear.x;
    ego_.steering = measured.tire_angle_front; ego_.x = measured.pose.pose.position.x;
    ego_.y = measured.pose.pose.position.y; ego_.z = measured.pose.pose.position.z;
    const auto & quaternion = measured.pose.pose.orientation;
    ego_.heading = pcm::yaw(quaternion);
    ego_.pitch = -std::asin(std::clamp(2 * (quaternion.w * quaternion.y - quaternion.z * quaternion.x), -1.0, 1.0));
    history_.push_back(ego_);
    auto scenario = std::make_shared<crp_msgs::msg::Scenario>();
    deserialize("scenario.cdr", *scenario);
    receive_scene(scenario);
    if (!scene_) {throw std::runtime_error("Probe scene was rejected");}
    Eigen::Vector4d actuator(ego_.speed * real("powertrain.motor_gear_ratio") / real("powertrain.wheel_radius"), 0.0, 0.0, 0.0);
    auto pedestrian = dynamic_gp(*scene_, true), vehicle = dynamic_gp(*scene_, false);
    auto inputs = pack(*scene_, ego_, actuator, pedestrian, vehicle);
    Eigen::VectorXd controls = Eigen::Map<const Eigen::VectorXd>(coefficients.data(), coefficients.size());
    auto report = reporting_(std::vector<Eigen::MatrixXd>{controls, inputs});
    Json result; result["parameters"] = flatten(inputs);
    std::vector<std::vector<double>> outputs;
    for (const auto & value : report) {outputs.push_back(flatten(value));}
    result["outputs"] = outputs;
    return result;
  }

private:
  double real(const std::string & name) const {return get_parameter(name).as_double();}
  int integer(const std::string & name) const {return static_cast<int>(get_parameter(name).as_int());}
  bool flag(const std::string & name) const {return get_parameter(name).as_bool();}
  Eigen::VectorXd parameters(const std::string & name) const
  {
    auto values = get_parameter(name).as_double_array();
    return Eigen::Map<Eigen::VectorXd>(values.data(), values.size());
  }
  static double seconds(Clock::time_point start) {return std::chrono::duration<double>(Clock::now() - start).count();}

  void receive_scene(crp_msgs::msg::Scenario::ConstSharedPtr message)
  {
    try {
      if (message->paths.empty()) {return;}
      auto scene = std::make_shared<Scene>(); scene->message = message;
      std::vector<Eigen::Vector3d> route, bounds, road;
      for (std::size_t path_index = 0; path_index < message->paths.size(); ++path_index) {
        const auto & path = message->paths[path_index];
        for (const auto & waypoint : path.path.points) {
          const auto & point = waypoint.point.pose.position;
          Eigen::Vector3d value(point.x, point.y, point.z);
          if (value.allFinite()) {(path_index ? road : route).push_back(value);}
        }
        if (!path_index) {
          scene->target_speed = path.traffic_rules.maximum_speed;
          for (const auto * boundary : {&path.path.left_bound, &path.path.right_bound}) {
            const int count = std::min(128, static_cast<int>(boundary->size()));
            for (int index = 0; index < count; ++index) {
              const auto & point = (*boundary)[count < 2 ? 0 : index * (boundary->size() - 1) / (count - 1)];
              bounds.emplace_back(point.x, point.y, point.z);
            }
          }
        }
      }
      if (route.size() < 2 || !std::isfinite(scene->target_speed) || scene->target_speed < 0) {return;}
      auto matrix = [](const std::vector<Eigen::Vector3d> & values) {
          Eigen::MatrixXd result(values.size(), 3);
          for (std::size_t index = 0; index < values.size(); ++index) {result.row(index) = values[index];}
          return result;
        };
      scene->route = matrix(route); scene->bounds = matrix(bounds);
      scene->reference = reference_->prepare(scene->route);
      scene->target = pcm::static_gp(scene->route, integer("target_support_points"), parameters("ego_lane_kernel"),
                                    real("ego_lane_cost"), real("lane_noise_std"));
      scene->road = pcm::static_gp(matrix(road), integer("drivable_support_points"), parameters("opposite_lane_kernel"),
                                  real("opposite_lane_cost"), real("drivable_noise_std"));
      std::lock_guard<std::mutex> lock(mutex_);
      if (!history_.empty()) {
        const auto stamp = rclcpp::Time(message->header.stamp).nanoseconds();
        scene->origin = *std::min_element(history_.begin(), history_.end(), [stamp](const auto & left, const auto & right) {
            return std::abs(left.stamp - stamp) < std::abs(right.stamp - stamp);});
        scene->origin_valid = true;
      }
      scene_ = scene;
    } catch (const std::exception & error) {RCLCPP_ERROR(get_logger(), "Scenario rejected: %s", error.what());}
  }

  pcm::GP dynamic_gp(const Scene & scene, bool pedestrian)
  {
    const auto kernel_parameters = parameters(pedestrian ? "pedestrian_kernel" : "vehicle_kernel");
    const int maximum = integer(pedestrian ? "max_pedestrians" : "max_vehicles");
    const int count = maximum * integer("max_predicted_paths") * steps_;
    const int capacity = pedestrian ? pedestrian_count_ : vehicle_count_;
    pcm::GP result{Eigen::MatrixXd::Constant(count, 3, 1000.0), Eigen::VectorXd::Zero(count)};
    Eigen::VectorXd targets = Eigen::VectorXd::Zero(count), noise = Eigen::VectorXd::Zero(count);
    const double base = real(pedestrian ? "pedestrian_noise_base" : "vehicle_noise_base");
    const double growth = real(pedestrian ? "pedestrian_noise_growth" : "vehicle_noise_growth");
    const double severity = real(pedestrian ? "pedestrian_cost" : "vehicle_cost");
    noise.setConstant(base);
    std::vector<std::vector<int>> groups;
    int objects = 0, used = 0;
    for (const auto & object : scene.message->local_moving_objects.objects) {
      const auto & pose = object.kinematics.initial_pose_with_covariance.pose;
      if (object.classification.empty() || (object.classification.front().label == 7) != pedestrian ||
          std::hypot(pose.position.x, pose.position.y) >= 150 || pose.position.x < -150 ||
          !std::isfinite(object.existence_probability) || object.existence_probability <= 0) {continue;}
      std::vector<std::size_t> order;
      const auto & predictions = object.kinematics.predicted_paths;
      for (std::size_t index = 0; index < predictions.size(); ++index) {
        const bool finite = std::all_of(predictions[index].path.begin(), predictions[index].path.end(), [](const auto & step) {
          return std::isfinite(step.position.x) && std::isfinite(step.position.y);});
        if (std::isfinite(predictions[index].confidence) && predictions[index].confidence > 0 && !predictions[index].path.empty() && finite) {
          order.push_back(index);
        }
      }
      std::stable_sort(order.begin(), order.end(), [&](auto left, auto right) {return predictions[left].confidence > predictions[right].confidence;});
      if (!predictions.empty() && order.empty()) {continue;}
      if (objects++ >= maximum) {break;}
      if (predictions.empty()) {order.push_back(0);}
      for (std::size_t branch = 0; branch < std::min(order.size(), static_cast<std::size_t>(integer("max_predicted_paths"))); ++branch) {
        const double confidence = predictions.empty() ? 1.0 : std::clamp(double(predictions[order[branch]].confidence), 0.0, 1.0);
        const double probability = std::clamp(double(object.existence_probability), 0.0, 1.0) * confidence;
        std::vector<double> times, position_x, position_y;
        if (!predictions.empty()) {
          const auto & prediction = predictions[order[branch]];
          double interval = prediction.time_step.sec + prediction.time_step.nanosec * 1e-9;
          if (interval <= 0) {interval = horizon_ / steps_;}
          for (std::size_t index = 0; index < prediction.path.size(); ++index) {
            times.push_back(index * interval); position_x.push_back(prediction.path[index].position.x);
            position_y.push_back(prediction.path[index].position.y);
          }
        }
        std::vector<int> group;
        for (int index = 0; index < steps_; ++index) {
          const double time = horizon_ * (index + 1) / steps_;
          const double speed = object.kinematics.initial_twist_with_covariance.twist.linear.x;
          double position_forward = pose.position.x + speed * std::cos(pcm::yaw(pose.orientation)) * time;
          double position_lateral = pose.position.y + speed * std::sin(pcm::yaw(pose.orientation)) * time;
          if (!times.empty()) {position_forward = pcm::interpolate(times, position_x, time); position_lateral = pcm::interpolate(times, position_y, time);}
          result.points.row(used) << position_forward, position_lateral, time;
          targets[used] = 1.0 + probability * severity;
          noise[used] = base + growth * time + probability * (1.0 - probability) * severity * severity;
          group.push_back(used++);
        }
        groups.push_back(group);
      }
    }
    auto matrix = pcm::kernel(result.points, result.points, kernel_parameters);
    result.alpha = pcm::fit(matrix, targets, noise);
    if (!flag("sparse_gp")) {return result;}
    auto projection = pcm::compress(result.points, result.alpha, kernel_parameters, groups, integer("object_support_points"), matrix);
    const double floor = -1.0 + 0.5 * (projection.error_bound + std::sqrt(projection.error_bound * projection.error_bound + 0.01));
    if (exact_gp_capacity_ && floor > real("max_gp_cost")) {return result;}
    pcm::GP reduced{Eigen::MatrixXd::Constant(capacity, 3, 1000.0), Eigen::VectorXd::Zero(capacity), projection.error_bound};
    for (std::size_t index = 0; index < projection.selected.size(); ++index) {
      reduced.points.row(index) = result.points.row(projection.selected[index]); reduced.alpha[index] = projection.coefficients[index];
    }
    return reduced;
  }

  Eigen::VectorXd pack(const Scene & scene, const EgoSample & ego, const Eigen::Vector4d & actuator,
                       const pcm::GP & pedestrian, const pcm::GP & vehicle)
  {
    std::vector<double> values(13, 0.0); values[0] = ego.speed;
    auto append = [&](const Eigen::MatrixXd & matrix) {auto entries = flatten(matrix); values.insert(values.end(), entries.begin(), entries.end());};
    append(pedestrian.alpha); append(pedestrian.points); append(vehicle.alpha); append(vehicle.points);
    append(scene.target.points); append(scene.target.alpha); append(scene.road.points); append(scene.road.alpha);
    Eigen::Vector4d frame = Eigen::Vector4d::Zero(); Eigen::VectorXd reference = scene.reference;
    if (actuators_ && scene.origin_valid) {
      const double delta_x = ego.x - scene.origin.x, delta_y = ego.y - scene.origin.y;
      frame[0] = std::cos(scene.origin.heading) * delta_x + std::sin(scene.origin.heading) * delta_y;
      frame[1] = -std::sin(scene.origin.heading) * delta_x + std::cos(scene.origin.heading) * delta_y;
      frame[2] = ego.heading - scene.origin.heading;
      frame[3] = std::max(0.0, (ego.stamp - rclcpp::Time(scene.message->header.stamp).nanoseconds()) * 1e-9);
      Eigen::MatrixXd route = scene.route;
      for (int index = 0; index < route.rows(); ++index) {
        const double forward = route(index, 0) - frame[0], lateral = route(index, 1) - frame[1];
        route(index, 0) = std::cos(frame[2]) * forward + std::sin(frame[2]) * lateral;
        route(index, 1) = -std::sin(frame[2]) * forward + std::cos(frame[2]) * lateral;
      }
      reference = reference_->prepare(route);
    }
    append(reference);
    const std::vector<double> context{ego.lateral, ego.yaw_rate, ego.pitch, ego.acceleration, ego.steering,
                                    ego.x, ego.y, ego.z, ego.heading, 0.0, 0.0, 0.0, 0.0};
    values.insert(values.end(), context.begin(), context.end());
    if (actuators_) {append(frame); append(actuator);}
    if (flag("sparse_gp")) {values.push_back(pedestrian.error); values.push_back(vehicle.error);}
    values.push_back(scene.target_speed);
    if (static_cast<int>(values.size()) != parameter_count_) {throw std::runtime_error("Native parameter layout mismatch");}
    return Eigen::Map<Eigen::VectorXd>(values.data(), values.size());
  }

  void plan()
  {
    const auto started = Clock::now();
    std::shared_ptr<Scene> scene; EgoSample ego; Eigen::Vector4d actuator;
    {
      std::lock_guard<std::mutex> lock(mutex_); scene = scene_; ego = ego_; actuator = powertrain_;
      if (!scene || (actuators_ && seconds(actuator_time_) > real("powertrain_state_timeout"))) {return;}
    }
    try {
      auto pedestrian = dynamic_gp(*scene, true), vehicle = dynamic_gp(*scene, false);
      Eigen::VectorXd inputs = pack(*scene, ego, actuator, pedestrian, vehicle);
      if (staged_) {
        plan_stages(*scene, ego, actuator, pedestrian, vehicle, inputs, started);
        return;
      }
      const auto initialized = Clock::now(); Eigen::VectorXd initial = previous_; bool warm = false;
      if (decisions_ == 4) {
        const double delta = previous_valid_ ? std::min(seconds(previous_time_), horizon_) : 0.0;
        if (previous_valid_) {
          const double fraction = delta / horizon_;
          for (int order = 1; order <= 4; ++order) {
            initial[order - 1] = 0.0;
            for (int power = order; power <= 4; ++power) {
              double binomial = 1.0;
              for (int factor = 1; factor <= order; ++factor) {binomial *= double(power - factor + 1) / factor;}
              initial[order - 1] += previous_[power - 1] * binomial * std::pow(fraction, power - order);
            }
          }
          auto checked = fg_(std::vector<Eigen::MatrixXd>{initial, inputs});
          const double normalized = real("command_lookahead_time") / horizon_;
          double command = ego.speed;
          for (int power = 0; power < 4; ++power) {command += initial[power] * std::pow(normalized, power + 1);}
          warm = pcm::eigen(checked[1]).maxCoeff() <= 0.05 && (!actuators_ || command > 1e-6);
        }
        if (!warm) {
          const int count = 80; Eigen::MatrixXd basis(count, 4);
          for (int index = 0; index < count; ++index) {
            for (int power = 0; power < 4; ++power) {basis(index, power) = std::pow(double(index + 1) / count, power + 1);}
          }
          auto decomposition = basis.colPivHouseholderQr();
          std::vector<Eigen::VectorXd> candidates;
          if (previous_valid_) {candidates.push_back(initial);}
          for (double rate : {0.0, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, 0.5, 1.0, 2.0}) {
            Eigen::VectorXd profile(count);
            for (int index = 0; index < count; ++index) {
              profile[index] = std::clamp(std::max(0.0, ego.speed) + rate * horizon_ * (index + 1) / count,
                                         0.0, std::max(scene->target_speed, std::max(0.0, ego.speed))) - std::max(0.0, ego.speed);
            }
            candidates.push_back(decomposition.solve(profile));
          }
          std::pair<double, double> best{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
          for (const auto & candidate : candidates) {
            auto evaluated = fg_(std::vector<Eigen::MatrixXd>{candidate, inputs});
            std::pair<double, double> score{std::max(0.0, evaluated[1].maxCoeff()), evaluated[0](0, 0)};
            if (score < best) {best = score; initial = candidate;}
          }
        }
      }
      const double initialization_seconds = seconds(initialized);
      const auto solve_started = Clock::now();
      pcm::SolverResult solution;
      bool used_retry = false;
      if (native_sqp_) {
        solution = optimizer_->solve(initial, inputs);
        if (!solution.success) {
          used_retry = true;
          auto retry = retry_.solve(solution.coefficients.allFinite() ? solution.coefficients : initial, inputs);
          solution = {retry.coefficients, retry.constraints, retry.objective, retry.success, retry.status, retry.iterations};
        }
      } else {
        auto result = (warm ? warm_ : primary_).solve(initial, inputs);
        solution = {result.coefficients, result.constraints, result.objective, result.success, result.status, result.iterations};
      }
      Json solved{{"x", flatten(solution.coefficients)}, {"g", flatten(solution.constraints)},
                  {"f", solution.objective}, {"success", solution.success},
                  {"status", solution.status}, {"iterations", solution.iterations}};
      const double solve_seconds = seconds(solve_started);
      auto coefficients = solved["x"].get<std::vector<double>>();
      auto constraints = solved["g"].get<std::vector<double>>();
      const double maximum = *std::max_element(constraints.begin(), constraints.end());
      bool valid = solved["success"].get<bool>() && coefficients.size() == static_cast<std::size_t>(decisions_) &&
        std::isfinite(solved["f"].get<double>()) && std::all_of(coefficients.begin(), coefficients.end(), [](double value) {return std::isfinite(value);}) &&
        std::all_of(constraints.begin(), constraints.end(), [](double value) {return std::isfinite(value);}) && maximum <= std::max(1e-6, 2 * real("solver_tolerance"));
      Json diagnostics;
      diagnostics["status"] = solved["status"]; diagnostics["accepted"] = valid; diagnostics["maximum_constraint"] = maximum;
      diagnostics["prediction_model"] = get_parameter("prediction_model").as_string();
      diagnostics["ego_speed"] = ego.speed; diagnostics["iterations"] = solved["iterations"];
      if (valid) {
        previous_ = Eigen::Map<Eigen::VectorXd>(coefficients.data(), coefficients.size()); previous_valid_ = true; previous_time_ = started;
        auto report = reporting_(std::vector<Eigen::MatrixXd>{previous_, inputs});
        publish(report, *scene, ego);
        if (actuators_) {
          Json energy; energy["drawn_j"] = report[10](0, 0); energy["recovered_j"] = report[11](0, 0);
          energy["net_j"] = report[12](0, 0); diagnostics["energy"] = energy;
        }
        if (flag("publish_gp_state") && seconds(gp_time_) >= real("gp_state_period")) {
          publish_gp(*scene, pedestrian, vehicle, diagnostics); gp_time_ = Clock::now();
        }
      } else {RCLCPP_WARN(get_logger(), "Native plan rejected: %s", solved["status"].get<std::string>().c_str());}
      if (flag("timing_enabled")) {
        diagnostics["total_seconds"] = seconds(started); diagnostics["optimizer_seconds"] = solve_seconds;
        diagnostics["initialization_seconds"] = initialization_seconds; diagnostics["deadline_seconds"] = period_;
        diagnostics["deadline_missed"] = seconds(started) > period_; diagnostics["implementation"] = "cpp_native_solver";
        diagnostics["solver_backend"] = used_retry ? "ipopt_retry" : (native_sqp_ ? "nlopt_slsqp" : "ipopt");
        std_msgs::msg::String timing; timing.data = diagnostics.dump(); timing_pub_->publish(timing);
      }
    } catch (const std::exception & error) {RCLCPP_ERROR(get_logger(), "Native planning failed: %s", error.what());}
  }

  void plan_stages(const Scene & scene, const EgoSample & ego, const Eigen::Vector4d & actuator,
                    const pcm::GP & pedestrian, const pcm::GP & vehicle, const Eigen::VectorXd & parameters,
                    Clock::time_point started)
  {
    const double duration = horizon_ / steps_;
    const double upper = std::max(ego.speed, scene.target_speed);
    Eigen::VectorXd initial = Eigen::VectorXd::Zero(15);
    initial[4] = ego.speed; initial[5] = ego.lateral; initial[6] = ego.yaw_rate; initial.segment(7, 4) = actuator;
    Eigen::VectorXd commands = Eigen::VectorXd::Constant(steps_, std::clamp(ego.speed, 0.0, upper));
    if (previous_valid_) {
      const double offset = std::min(seconds(previous_time_), horizon_) / duration;
      for (int index = 0; index < steps_; ++index) {
        const double location = std::min(index + offset, double(steps_ - 1));
        int lower = static_cast<int>(std::floor(location)), next = std::min(lower + 1, steps_ - 1);
        commands[index] = std::clamp(previous_[lower] + (location - lower) * (previous_[next] - previous_[lower]), 0.0, upper);
      }
    }
    Eigen::MatrixXd guesses(15, steps_ + 1);
    guesses.col(0) = initial;
    for (int index = 0; index < steps_; ++index) {
      guesses.col(index + 1) = stage_({guesses.col(index), pcm::scalar(commands[index]), parameters})[0];
    }
    const auto solve_started = Clock::now();
    auto solution = stage_solver_->solve(parameters, initial, commands, guesses);
    const double solve_seconds = seconds(solve_started);
    const double tolerance = std::max(1e-6, 2 * real("solver_tolerance"));
    bool valid = solution.status == 0 && solution.commands.allFinite() && solution.states.allFinite() &&
      (solution.states.col(0) - initial).cwiseAbs().maxCoeff() <= tolerance;
    double maximum = -std::numeric_limits<double>::infinity();
    std::vector<Eigen::MatrixXd> report(15);
    report[0] = Eigen::MatrixXd::Zero(steps_, 3); report[1] = Eigen::MatrixXd::Zero(steps_, 1); report[2] = Eigen::MatrixXd::Zero(steps_, 3);
    report[3] = Eigen::MatrixXd::Zero(steps_, 1); report[4] = pcm::scalar(solution.commands[0]);
    report[5] = Eigen::MatrixXd::Zero(steps_, 1); report[6] = Eigen::MatrixXd::Zero(steps_, 1);
    report[7] = Eigen::MatrixXd::Zero(steps_, 4); report[8] = Eigen::MatrixXd::Zero(steps_, 7); report[9] = Eigen::MatrixXd::Zero(steps_, 1);
    report[13] = Eigen::MatrixXd::Zero(steps_, 6); report[14] = solution.commands;
    Eigen::VectorXd replay = initial;
    double objective = 0.0;
    if (solution.commands.allFinite() && solution.states.allFinite()) {
      for (int index = 0; index < steps_; ++index) {
        auto evaluated = stage_({replay, pcm::scalar(solution.commands[index]), parameters});
        auto expected = stage_({solution.states.col(index), pcm::scalar(solution.commands[index]), parameters})[0];
        maximum = std::max(maximum, evaluated[6].maxCoeff());
        valid = valid && evaluated[0].allFinite() && evaluated[6].allFinite() &&
          evaluated[6].maxCoeff() <= tolerance && solution.commands[index] >= -tolerance &&
          solution.commands[index] <= upper + tolerance &&
          (expected - solution.states.col(index + 1)).cwiseAbs().maxCoeff() <= tolerance;
        replay = evaluated[0]; objective += evaluated[7](0, 0);
        report[0].row(index) = evaluated[4].transpose(); report[1](index, 0) = replay[4];
        report[2].row(index) = evaluated[1].transpose(); report[5](index, 0) = evaluated[3](0, 0);
        report[7].row(index) = evaluated[5].transpose(); report[8].row(index) = replay.head(7).transpose();
        report[9](index, 0) = replay[13]; report[13].row(index) = replay.segment(7, 6).transpose();
      }
    } else {maximum = std::numeric_limits<double>::infinity();}
    valid = valid && std::isfinite(objective);
    report[10] = pcm::scalar(replay[11]); report[11] = pcm::scalar(replay[12]); report[12] = pcm::scalar(replay[11] - replay[12]);
    Json diagnostics{{"status", valid ? "Solve_Succeeded" : "acados_stage_status_" + std::to_string(solution.status)},
      {"accepted", valid}, {"prediction_model", get_parameter("prediction_model").as_string()},
      {"velocity_representation", "acados_stages"}, {"solver_backend", "acados_sqp"},
      {"maximum_constraint", std::isfinite(maximum) ? Json(maximum) : Json(nullptr)},
      {"ego_speed", ego.speed}, {"iterations", solution.iterations}, {"objective", objective},
      {"optimizer_seconds", solve_seconds}, {"total_seconds", seconds(started)},
      {"deadline_seconds", period_}, {"deadline_missed", seconds(started) > period_},
      {"speed_command_points", flatten(solution.commands)}, {"predicted_velocity_points", flatten(report[1])}};
      diagnostics["energy"] = {{"drawn_j", replay[11]}, {"recovered_j", replay[12]},
                   {"net_j", replay[11] - replay[12]}};
    if (valid) {
      previous_ = solution.commands; previous_valid_ = true; previous_time_ = started;
      publish(report, scene, ego);
      if (flag("publish_gp_state") && seconds(gp_time_) >= real("gp_state_period")) {
        publish_gp(scene, pedestrian, vehicle, diagnostics); gp_time_ = Clock::now();
      }
    } else {RCLCPP_WARN(get_logger(), "Stage-wise acados plan rejected: status=%d maximum=%g", solution.status, maximum);}
    if (flag("timing_enabled")) {
      diagnostics["total_seconds"] = seconds(started);
      std_msgs::msg::String message; message.data = diagnostics.dump(); timing_pub_->publish(message);
    }
  }

  void publish(const std::vector<Eigen::MatrixXd> & report, const Scene & scene, const EgoSample & planned_ego)
  {
    autoware_control_msgs::msg::Control command; command.stamp = now(); command.longitudinal.velocity = std::max(0.0, report[4](0, 0));
    command_pub_->publish(command);
    auto positions = pcm::eigen(report[0]), speed = pcm::eigen(report[1]), acceleration = pcm::eigen(report[2]);
    auto steering = pcm::eigen(report[5]), orientation = pcm::eigen(report[7]), states = pcm::eigen(report[8]);
    EgoSample current;
    {std::lock_guard<std::mutex> lock(mutex_); current = ego_;}
    const auto & origin = actuators_ ? planned_ego : scene.origin;
    const double heading = origin.heading - current.heading;
    const double shift_x = std::cos(current.heading) * (origin.x - current.x) + std::sin(current.heading) * (origin.y - current.y);
    const double shift_y = -std::sin(current.heading) * (origin.x - current.x) + std::cos(current.heading) * (origin.y - current.y);
    autoware_planning_msgs::msg::Trajectory trajectory; trajectory.header.frame_id = "base_link"; trajectory.header.stamp = now();
    for (int index = 0; index < steps_; ++index) {
      autoware_planning_msgs::msg::TrajectoryPoint point;
      point.pose.position.x = shift_x + std::cos(heading) * positions(index, 0) - std::sin(heading) * positions(index, 1);
      point.pose.position.y = shift_y + std::sin(heading) * positions(index, 0) + std::cos(heading) * positions(index, 1);
      point.pose.position.z = positions(index, 2);
      const double sine = std::sin(heading / 2), cosine = std::cos(heading / 2);
      point.pose.orientation.x = cosine * orientation(index, 0) - sine * orientation(index, 1);
      point.pose.orientation.y = cosine * orientation(index, 1) + sine * orientation(index, 0);
      point.pose.orientation.z = cosine * orientation(index, 2) + sine * orientation(index, 3);
      point.pose.orientation.w = cosine * orientation(index, 3) - sine * orientation(index, 2);
      point.longitudinal_velocity_mps = speed(index, 0); point.acceleration_mps2 = acceleration(index, 0);
      point.lateral_velocity_mps = states(index, 5); point.heading_rate_rps = states(index, 6);
      point.front_wheel_angle_rad = steering(index, 0); point.time_from_start = rclcpp::Duration::from_seconds(horizon_ * (index + 1) / steps_);
      trajectory.points.push_back(point);
    }
    trajectory_pub_->publish(trajectory);
  }

  void publish_gp(const Scene & scene, const pcm::GP & pedestrian, const pcm::GP & vehicle, const Json & diagnostics)
  {
    Json layers;
    auto layer = [&](const std::string & name, const pcm::GP & gp, const std::string & kernel_name, double offset) {
        Json value; value["points"] = rows(gp.points); value["alpha"] = flatten(gp.alpha);
        value["kernel"] = flatten(parameters(kernel_name)); value["offset"] = offset; layers[name] = value;
      };
    layer("pedestrian", pedestrian, "pedestrian_kernel", -1.0 + pedestrian.error);
    layer("vehicle", vehicle, "vehicle_kernel", -1.0 + vehicle.error);
    layer("ego_lane", scene.target, "ego_lane_kernel", 0.0); layer("opposite_lane", scene.road, "opposite_lane_kernel", 0.0);
    std::vector<double> times;
    for (int index = 0; index < steps_; ++index) {times.push_back(horizon_ * (index + 1) / steps_);}
    Json snapshot; snapshot["version"] = 2; snapshot["frame_id"] = "base_link";
    snapshot["stamp_ns"] = rclcpp::Time(scene.message->header.stamp).nanoseconds(); snapshot["planning"] = diagnostics;
    snapshot["times"] = times; snapshot["road"] = rows(scene.route); snapshot["road_bounds"] = rows(scene.bounds);
    snapshot["layers"] = layers; snapshot["progress_reference"] = flatten(scene.reference);
    if (staged_) {
      snapshot["velocity_representation"] = "acados_stages";
      snapshot["speed_command_points"] = flatten(previous_);
      if (diagnostics.contains("predicted_velocity_points")) {
        snapshot["predicted_velocity_points"] = diagnostics["predicted_velocity_points"];
      }
    } else {snapshot["optimized_coefficients"] = flatten(previous_);}
    std_msgs::msg::String message; message.data = snapshot.dump(); gp_pub_->publish(message);
  }

  bool actuators_ = false, previous_valid_ = false;
  bool exact_gp_capacity_ = false;
  int decisions_, parameter_count_, prediction_outputs_, pedestrian_count_, vehicle_count_, steps_;
  double horizon_, period_;
  Eigen::VectorXd previous_; Eigen::Vector4d powertrain_ = Eigen::Vector4d::Zero();
  Clock::time_point previous_time_{}, actuator_time_{}, gp_time_{};
  EgoSample ego_; std::deque<EgoSample> history_; std::shared_ptr<Scene> scene_; std::mutex mutex_;
  pcm::Function fg_, reporting_, primary_, warm_, retry_;
  std::unique_ptr<pcm::Reference> reference_; std::unique_ptr<pcm::NativeSLSQP> optimizer_;
  bool native_sqp_ = false;
  bool staged_ = false;
  pcm::Function stage_;
  std::unique_ptr<pcm::AcadosStages> stage_solver_;
  rclcpp::CallbackGroup::SharedPtr telemetry_group_, scene_group_;
  rclcpp::Subscription<crp_msgs::msg::Ego>::SharedPtr ego_sub_;
  rclcpp::Subscription<crp_msgs::msg::ActuatorState>::SharedPtr actuator_sub_;
  rclcpp::Subscription<crp_msgs::msg::Scenario>::SharedPtr scenario_sub_;
  rclcpp::Publisher<autoware_control_msgs::msg::Control>::SharedPtr command_pub_;
  rclcpp::Publisher<autoware_planning_msgs::msg::Trajectory>::SharedPtr trajectory_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr timing_pub_, gp_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  if (argc == 3 && std::string(argv[1]) == "--model-probe") {
    std::string encoded((std::istreambuf_iterator<char>(std::cin)), {});
    Json request = Json::parse(encoded);
    rclcpp::init(0, nullptr);
    rclcpp::NodeOptions options;
    options.parameter_overrides({rclcpp::Parameter("model_directory", std::string(argv[2]))});
    auto node = std::make_shared<NativePlanner>(options);
    auto result = node->model_probe(argv[2], request["coefficients"].get<std::vector<double>>());
    std::cout << "PARITY " << result.dump() << '\n';
    node.reset(); rclcpp::shutdown(); return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--preprocess-probe") {
    std::string encoded((std::istreambuf_iterator<char>(std::cin)), {});
    Json request = Json::parse(encoded);
    auto input = request["route"].get<std::vector<std::vector<double>>>();
    Eigen::MatrixXd route(input.size(), 3);
    for (std::size_t index = 0; index < input.size(); ++index) {
      for (int axis = 0; axis < 3; ++axis) {route(index, axis) = input[index][axis];}
    }
    auto kernel_values = request["kernel"].get<std::vector<double>>();
    Eigen::VectorXd kernel = Eigen::Map<Eigen::VectorXd>(kernel_values.data(), kernel_values.size());
    pcm::Reference reference(pcm::Function::load(std::string(argv[2]) + "/basis.casadi"), request["samples"].get<int>());
    auto gp = pcm::static_gp(route, request["supports"].get<int>(), kernel,
                  request["cost"].get<double>(), request["noise"].get<double>());
    Json result; result["reference"] = flatten(reference.prepare(route)); result["points"] = rows(gp.points);
    result["alpha"] = flatten(gp.alpha);
    std::cout << result.dump() << '\n';
    return 0;
  }
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<NativePlanner>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
    executor.add_node(node);
    executor.spin();
    executor.remove_node(node); node.reset();
  } catch (const std::exception & error) {std::cerr << error.what() << '\n'; rclcpp::shutdown(); return 1;}
  rclcpp::shutdown(); return 0;
}