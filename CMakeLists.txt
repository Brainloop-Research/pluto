# (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

cmake_minimum_required(VERSION 3.10)

project(pluto LANGUAGES C)

set(CMAKE_C_STANDARD 99)

option(PLUTO_BUILD_TESTS "Build tests" ON)

file(GLOB_RECURSE SOURCES include/*.h src/*.c)

if (NOT WIN32)
    add_compile_options(-gdwarf-4)
endif()

add_library(pluto STATIC ${SOURCES})
target_include_directories(pluto PUBLIC include/pluto)
target_compile_options(
    pluto
    PRIVATE
    -Wall
    -Wextra
    -Werror
    -pedantic
    -Wno-gnu-zero-variadic-macro-arguments
)

if (PLUTO_BUILD_TESTS)
    add_subdirectory(test)
endif()
