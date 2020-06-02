#  Author: Arjun Ramaswami

cmake_minimum_required(VERSION 3.10)

add_subdirectory("${PROJECT_SOURCE_DIR}/extern/gtest" "extern/gtest")

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

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "../extern/findFFTW")

macro(package_add_test TESTNAME FILES LIBRARIES INCLUDE DEFN TEST_WORKING_DIR)
    message("${FILES}")
    message("${LIBRARIES}")
    message("${INCLUDE}")
    message("${DEFN}")
    
    add_executable(${TESTNAME} ${FILES})

    target_include_directories(${TESTNAME}
    PRIVATE
        ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR}
        ${IntelFPGAOpenCL_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/extern/argparse 
        ${INCLUDE})

    target_compile_definitions(${TESTNAME} PRIVATE ${DEFN})
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME}
      PUBLIC gtest gmock gtest_main 
      ${IntelFPGAOpenCL_LIBRARIES} 
      ${LIBRARIES}
    )
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${TEST_WORKING_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${TEST_WORKING_DIR}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
endmacro()

set(FILES
  ${CMAKE_SOURCE_DIR}/extern/argparse/argparse.c
  ${PROJECT_SOURCE_DIR}/api/src/opencl_utils.c
  ${PROJECT_SOURCE_DIR}/api/src/fftfpga.c 
  ${PROJECT_SOURCE_DIR}/api/src/helper.c

  ${CMAKE_SOURCE_DIR}/tests/test_opencl_utils.cpp
  ${CMAKE_SOURCE_DIR}/tests/test_helper.cpp
  ${CMAKE_SOURCE_DIR}/tests/test_fft_fpga.cpp
)

set(INCL ${CMAKE_SOURCE_DIR}/api/include)
set(LIBS fftw3f fftw3 m)

find_package(FFTW)
if(FFTW_FOUND)
  package_add_test(test1 "${FILES}" "${LIBS}" "${INCL}" USE_FFTW ${PROJECT_SOURCE_DIR})
  message("-- FFTW library found")
else()
  message(WARNING, "FFTW library not found. Cannot perform correctness tests!")
endif()


