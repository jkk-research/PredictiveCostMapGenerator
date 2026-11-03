#!/bin/bash
nproc
grep -a 'Loop time' /tmp/live_pcm_kinematic.log | sed -E 's/.*Loop time: ([0-9.]+)s/\1/' | sort -n \
  | awk '{a[NR]=$1} END {print "loops", NR, "median", a[int(NR/2)], "p95", a[int(NR*0.95)], "max", a[NR]}'
grep -nE 'create_wall_timer|create_timer|period|QoS|rclcpp::QoS|create_publisher' \
  /home/igg2bp/ros2_ws/src/CooperativeResearchPlatform_Simulator/scenario_generation/src/scenario_generator_node.cpp | head -20
