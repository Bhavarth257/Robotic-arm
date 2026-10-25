cmake_minimum_required(VERSION 3.8)
project(so101_description)

find_package(ament_cmake REQUIRED)

# Single source of truth: the simulation asset. Only mesh paths change, to ROS package URIs.
set(ASSET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../assets/so101)
file(READ ${ASSET_DIR}/so101.urdf URDF)
string(REPLACE "filename=\"meshes/" "filename=\"package://${PROJECT_NAME}/meshes/" URDF "${URDF}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/so101.urdf "${URDF}")
# Re-run this step whenever the asset URDF changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ASSET_DIR}/so101.urdf)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/so101.urdf DESTINATION share/${PROJECT_NAME}/urdf)
install(DIRECTORY ${ASSET_DIR}/meshes DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch rviz DESTINATION share/${PROJECT_NAME})

ament_package()
