cmake_minimum_required(VERSION 3.8)
project(so101_hardware)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(${PROJECT_NAME} SHARED src/sts_bus.cpp src/so101_system.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)
ament_target_dependencies(${PROJECT_NAME} hardware_interface pluginlib rclcpp rclcpp_lifecycle)
target_link_libraries(${PROJECT_NAME} yaml-cpp)
pluginlib_export_plugin_description_file(hardware_interface so101_hardware.xml)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
install(DIRECTORY config launch urdf DESTINATION share/${PROJECT_NAME})

ament_package()
