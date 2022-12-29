cmake_minimum_required(VERSION 3.20)
project(mechanics)

set(CMAKE_CXX_STANDARD 20)
find_package(Eigen3 CONFIG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)

add_executable(mechanics main.cpp)
target_link_libraries(mechanics PRIVATE Eigen3::Eigen )
IF (WIN32)
target_link_libraries(mechanics PRIVATE jsoncpp_lib jsoncpp_object JsonCpp::JsonCpp)
ELSE()
target_link_libraries(mechanics PRIVATE jsoncpp_object jsoncpp_static JsonCpp::JsonCpp)
ENDIF()
