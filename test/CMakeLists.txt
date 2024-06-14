# (c) 2024 Brainloop Research. <mario.sieg.64@gmail.com>

enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)
enable_testing()

file(GLOB_RECURSE TEST_SOURCES src/*.hpp src/*.cpp)

add_subdirectory(googletest)

add_executable(pluto_test ${TEST_SOURCES})
target_link_libraries(pluto_test pluto)
target_include_directories(pluto_test PUBLIC ../include/pluto)
target_include_directories(pluto_test PUBLIC ../googletest/googletest/include)
target_link_libraries(pluto_test gtest_main)
