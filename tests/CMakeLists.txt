#  Author: Arjun Ramaswami

cmake_minimum_required(VERSION 3.10)

add_subdirectory("${PROJECT_SOURCE_DIR}/extern/googletest" "extern/googletest")

# Cleaner Cache
mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

# IDEs that support clean folders ??
set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)
set_target_properties(gmock PROPERTIES FOLDER extern)
set_target_properties(gmock_main PROPERTIES FOLDER extern)

macro(package_add_test TESTNAME FILES LIBRARIES TEST_WORKING_DIRECTORY)
    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ARGN})
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME}
      PUBLIC gtest gmock gtest_main 
      ${IntelFPGAOpenCL_LIBRARIES}
    )
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

package_add_test(test1 test_opencl_utils.cpp)
package_add_test(test2 test_helper.cpp)
package_add_test(test3 test_fft_fpga.cpp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "../extern/findFFTW")

set(Sources_dir ${CMAKE_SOURCE_DIR}/src/host)

set(Sources 
  ${CMAKE_SOURCE_DIR}/extern/argparse/argparse.c
#  ${Sources_dir}/main.c
  ${Sources_dir}/src/opencl_utils.c
  ${Sources_dir}/src/fftfpga.c 
  ${Sources_dir}/src/helper.c
)

set(TestSources
  ${CMAKE_SOURCE_DIR}/tests/test_opencl_utils.cpp
  ${CMAKE_SOURCE_DIR}/tests/test_helper.cpp
  ${CMAKE_SOURCE_DIR}/tests/test_fft_fpga.cpp
)

add_executable(${This} ${Sources} ${TestSources})

find_package(FFTW)
if(FFTW_FOUND)
  target_compile_definitions(${This} PRIVATE USE_FFTW)
  target_link_libraries(${This} PUBLIC fftw3 fftw3f m)
  message("-- FFTW library found")
else()
  message(WARNING, "FFTW library not found. Cannot perform correctness tests!")
endif()

include_directories(
  ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/extern/argparse 
  ${CMAKE_SOURCE_DIR}/src/host/include
  ${IntelFPGAOpenCL_INCLUDE_DIRS}
)
