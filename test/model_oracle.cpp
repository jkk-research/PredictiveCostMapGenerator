#include "simulator/vehicle_model.hpp"
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char ** argv)
{
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