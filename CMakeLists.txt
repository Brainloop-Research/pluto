# (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

cmake_minimum_required(VERSION 3.10)

project(pluto LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)

option(PLUTO_BUILD_TESTS "Build tests" ON)

file(GLOB_RECURSE SOURCES src/*.h src/*.c src/*.cpp src/*.hpp)

if (WIN32)
   if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
   endif()
else()
    add_compile_options(-gdwarf-4)
endif()

add_library(pluto STATIC ${SOURCES})
target_include_directories(pluto PUBLIC include/pluto)
if (NOT WIN32)
    target_compile_options(
        pluto
        PRIVATE
        -Wall
        -Wextra
        -Werror
        -pedantic
        -Wno-gnu-zero-variadic-macro-arguments
        -Wno-gnu-statement-expression
    )
endif()

if (PLUTO_BUILD_TESTS)
    add_subdirectory(test)
endif()
