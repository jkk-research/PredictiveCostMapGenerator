#include "simulator/vehicle_model.hpp"
#ifdef PCM_POWERTRAIN_ORACLE
#include "simulator/powertrain_model.hpp"
#include <algorithm>
#include <cmath>
#endif
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char ** argv)
{
#ifdef PCM_POWERTRAIN_ORACLE
    if (argc == 9 && std::string(argv[1]) == "--actuated")
    {
        crp::apl::PowertrainModel powertrain(argv[2]);
        crp::apl::VehicleModel model;
        model.settings.model = std::string(argv[7]) == "dynamic" ?
            crp::apl::VehicleModelType::Dynamic : crp::apl::VehicleModelType::Kinematic;
        model.state.speed = std::stod(argv[3]);
        model.state.pitch = std::stod(argv[6]);
        const double initial_speed = model.state.speed;
        const double target_speed = std::stod(argv[4]);
        const double duration = std::stod(argv[5]);
        const double steering = std::stod(argv[8]);
        powertrain.initialize(initial_speed);
        double filtered = 0.0;
        const auto count = static_cast<std::size_t>(std::ceil(duration / powertrain.settings.integration_step));
        const double step = duration / count;
        for (std::size_t index = 0; index < count; ++index)
        {
            filtered += -std::expm1(-step / 0.05) * (model.accelerationTarget(target_speed) - filtered);
            const double force = powertrain.step(filtered, model.state.speed, step, model.settings);
            model.stepWheelForce(force, steering, step, model.state.pitch);
        }
        const auto & state = model.state;
        const auto & motor = powertrain.state;
        std::cout << std::setprecision(17) << state.x << ' ' << state.y << ' ' << state.z << ' '
            << state.yaw << ' ' << state.speed << ' ' << state.lateral_speed << ' ' << state.yaw_rate << ' '
            << motor.motor_speed << ' ' << motor.shaft_twist << ' ' << motor.motor_torque << ' '
            << filtered << ' ' << motor.energy_consumed << ' ' << motor.energy_regenerated << '\n';
        return 0;
    }
    if (argc == 10 && std::string(argv[1]) == "--powertrain")
    {
        crp::apl::PowertrainModel model(argv[2]);
        crp::apl::VehicleSettings vehicle;
        const double speed = std::stod(argv[3]);
        const double requested = std::stod(argv[4]);
        const double duration = std::stod(argv[5]);
        model.initialize(speed);
        model.state.motor_speed = std::stod(argv[6]);
        model.state.shaft_twist = std::stod(argv[7]);
        model.state.motor_torque = std::stod(argv[8]);
        double filtered = std::stod(argv[9]);
        const auto count = static_cast<std::size_t>(std::ceil(duration / model.settings.integration_step));
        const double step = duration / count;
        double integral = 0.0;
        for (std::size_t index = 0; index < count; ++index)
        {
            filtered += -std::expm1(-step / 0.05) * (requested - filtered);
            integral += model.step(filtered, speed, step, vehicle) * step;
        }
        const auto & state = model.state;
        std::cout << std::setprecision(17) << state.motor_speed << ' ' << state.shaft_twist << ' '
            << state.motor_torque << ' ' << filtered << ' ' << state.energy_consumed << ' '
            << state.energy_regenerated << ' ' << integral / duration << '\n';
        return 0;
    }
#endif
    if (argc == 3 && std::string(argv[1]) == "--follow")
    {
        crp::apl::VehicleModel follower;
        follower.state.x = 6.0; follower.state.y = 0.3; follower.state.z = 0.6;
        follower.state.yaw = 0.1; follower.state.pitch = 0.1;
        follower.state.speed = std::stod(argv[2]);
        const std::vector<crp::apl::MapPoint> path{{0.0, 0.0, 0.0}, {5.0, 0.0, 0.5},
            {10.0, 1.0, 1.0}, {20.0, 5.0, 2.0}, {40.0, 5.0, 3.0}};
        std::cout << std::setprecision(17) << follower.followPath(path, 1) << '\n';
        return 0;
    }
    if (argc != 12) {return 2;}
    crp::apl::VehicleModel model;
    model.settings.model = crp::apl::VehicleModelType::Dynamic;
    auto & state = model.state;
    state.x = std::stod(argv[1]); state.y = std::stod(argv[2]); state.z = std::stod(argv[3]);
    state.yaw = std::stod(argv[4]); state.speed = std::stod(argv[5]);
    state.lateral_speed = std::stod(argv[6]); state.yaw_rate = std::stod(argv[7]);
    state.pitch = std::stod(argv[10]);
    model.stepAcceleration(std::stod(argv[8]), std::stod(argv[9]), 0.05, std::stod(argv[11]));
    std::cout << std::setprecision(17) << state.x << ' ' << state.y << ' ' << state.z << ' '
        << state.yaw << ' ' << state.speed << ' ' << state.lateral_speed << ' ' << state.yaw_rate << ' '
        << state.acceleration << ' ' << state.lateral_acceleration << ' ' << state.yaw_acceleration << '\n';
}