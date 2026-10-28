cmake_minimum_required(VERSION 3.8)
project(so101_policy_bridge)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# Prebuilt ONNX Runtime C++ release, unpacked in <repo>/third_party (see PROGRESS M7c).
set(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
set(ONNXRUNTIME_ROOT ${REPO_ROOT}/third_party/onnxruntime-linux-x64-1.23.2 CACHE PATH "ONNX Runtime release dir")
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
  INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_ROOT}/include)

add_library(policy_core src/policy_core.cpp)
target_include_directories(policy_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries(policy_core onnxruntime)

add_executable(bridge_node src/bridge_node.cpp)
target_link_libraries(bridge_node policy_core)
ament_target_dependencies(bridge_node rclcpp sensor_msgs geometry_msgs std_msgs tf2_ros)

# Find libonnxruntime.so at run time without LD_LIBRARY_PATH.
set_target_properties(policy_core bridge_node PROPERTIES INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib)

install(TARGETS policy_core bridge_node
  LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES ${REPO_ROOT}/models/so101_pickplace.onnx ${REPO_ROOT}/models/so101_pickplace.json
  DESTINATION share/${PROJECT_NAME}/models)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_policy_core test/test_policy_core.cpp)
  target_link_libraries(test_policy_core policy_core)
  target_compile_definitions(test_policy_core PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data"
    MODEL_PATH="${REPO_ROOT}/models/so101_pickplace.onnx")
  set_target_properties(test_policy_core PROPERTIES BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib)
endif()

ament_package()
