cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_C_COMPILER "gcc")

project(RaBitQ LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(Green "${Esc}[32m")
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message("${Green}Release mode${ColourReset}")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message("${Green}Clang${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=core-avx2")
        add_definitions(-O3 -DNDEBUG -march=core-avx2)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message("${Green}GNU${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=core-avx2")
        add_definitions(-O3 -DNDEBUG -march=core-avx2)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message("${Green}MSVC${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=core-avx2")
        add_definitions(-O3 -DNDEBUG -march=core-avx2)
    else ()
        message("${Green}Unknown compiler${ColourReset}")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native -fsanitize=address -lasan ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=core-avx2 ${OpenMP_CXX_FLAGS}")
    #    add_definitions(-DNDEBUG -march=native)
    message("${Green}Debug mode${ColourReset}")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=core-avx2 ${OpenMP_CXX_FLAGS}")
    message("${Green}Unknown mode${ColourReset}")
endif ()


find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(Threads REQUIRED)

find_package(spdlog REQUIRED)

find_package(OpenMP REQUIRED)


add_executable(index_siftsmall src/index.cpp)
target_link_libraries(index_siftsmall PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen spdlog::spdlog)
target_include_directories(index_siftsmall PRIVATE src)
target_compile_definitions(index_siftsmall PRIVATE numC=4096 B_QUERY=4 SCAN)

add_executable(search_siftsmall src/search.cpp)
target_link_libraries(search_siftsmall PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen spdlog::spdlog)
target_include_directories(search_siftsmall PRIVATE src)
target_compile_definitions(search_siftsmall PRIVATE numC=4096 B_QUERY=4 SCAN)

add_executable(t1 test.cpp)
target_link_libraries(t1)


