cmake_minimum_required(VERSION 3.8)
project(so101_perception)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d aruco)

add_executable(aruco_node src/aruco_node.cpp)
target_link_libraries(aruco_node ${OpenCV_LIBS})
ament_target_dependencies(aruco_node rclcpp sensor_msgs geometry_msgs tf2 tf2_ros)

install(TARGETS aruco_node RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
