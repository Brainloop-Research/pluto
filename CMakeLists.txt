cmake_minimum_required(VERSION 3.10)

project(pluto LANGUAGES C)

set(CMAKE_C_STANDARD 99)

file(GLOB_RECURSE SOURCES include/*.h src/*.c)

add_executable(pluto ${SOURCES})
target_include_directories(pluto PUBLIC include/pluto)


