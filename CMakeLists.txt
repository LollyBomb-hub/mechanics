cmake_minimum_required(VERSION 3.18)
project(mechanics)

include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-march=native" _march_native_works)
check_cxx_compiler_flag("-xHost" _xhost_works)
check_cxx_compiler_flag("-mfma" _mfma)

set(_CXX_FLAGS)
if(_march_native_works)
    message(STATUS "Using processor's vector instructions (-march=native compiler flag set)")
    set(_CXX_FLAGS "-march=native")
elseif(_xhost_works)
    message(STATUS "Using processor's vector instructions (-xHost compiler flag set)")
    set(_CXX_FLAGS "-xHost")
else()
    message(STATUS "No suitable compiler flag found for vectorization")
endif()

if (_mfma)
    set(_FMA_FLAG "-mfma")
endif()

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

target_compile_options(
        mechanics
        PRIVATE
        ${_CXX_FLAGS}
        ${_FMA_FLAG}
        )