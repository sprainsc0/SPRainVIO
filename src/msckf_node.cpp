#include <cstdio>
#include <atomic>
#include <getopt.h>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <csignal>
#include <termios.h>
#include <memory>

#include <ekf_core/ekf_Core.h>

#include <rclcpp/rclcpp.hpp>

std::shared_ptr<ekf_Core>  _ekfcore;

int main(int argc, char ** argv)
{
  std::cout << "\033[1;33m--- SPRain MSCKF Navigation System ---\033[0m" << std::endl;

	rclcpp::init(argc, argv, rclcpp::InitOptions());

  auto node = std::make_shared<rclcpp::Node>("msckf_node");

  _ekfcore = std::make_shared<ekf_Core>(node);
  _ekfcore->setup_core();

  std::cout << "\033[1;33m--- SPRain MSCKF Navigation Start ---\033[0m" << std::endl;

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  std::cout << "\033[1;33m--- SPRain MSCKF Navigation Shutdown ---\033[0m" << std::endl;

  _ekfcore->shutdown();

  rclcpp::shutdown();

  return 0;
}
