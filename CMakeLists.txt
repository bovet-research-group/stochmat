cmake_minimum_required(VERSION 3.21...3.28)
project(stochmat LANGUAGES C CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Define common compile options for all Cython modules
add_library(stochmat_compile_options INTERFACE)
target_compile_options(stochmat_compile_options INTERFACE -O3)
target_compile_definitions(stochmat_compile_options INTERFACE NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

function(add_cython_cpp_module TARGET_NAME CYTHON_SOURCE)
    set(GENERATED_CXX_FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.cxx")
    
    get_filename_component(CYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${CYTHON_SOURCE}" DIRECTORY)
    
    # Set base Cython compiler flags
    set(CYTHON_FLAGS "-3" "--cplus")
    
    # Add coverage directives if CYTHON_COVERAGE env var is set
    if(DEFINED ENV{CYTHON_COVERAGE})
        list(APPEND CYTHON_FLAGS 
            "--directive" "linetrace=True"
            "--directive" "binding=True"
        )
        message(STATUS "Cython coverage enabled for ${TARGET_NAME}")
    endif()
    
    add_custom_command(
        OUTPUT "${GENERATED_CXX_FILE}"
        COMMAND "${Python_EXECUTABLE}" -m cython ${CYTHON_FLAGS} 
                "${CMAKE_CURRENT_SOURCE_DIR}/${CYTHON_SOURCE}" 
                -o "${GENERATED_CXX_FILE}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${CYTHON_SOURCE}"
        COMMENT "Generating C++ source from Cython definition: ${CYTHON_SOURCE}"
    )
    
    Python_add_library(${TARGET_NAME} MODULE WITH_SOABI "${GENERATED_CXX_FILE}")
    target_link_libraries(${TARGET_NAME} PRIVATE Python::NumPy stochmat_compile_options)
    
    # Add C compile-time macros for coverage if enabled
    if(DEFINED ENV{CYTHON_COVERAGE})
        target_compile_definitions(${TARGET_NAME} PRIVATE 
            CYTHON_TRACE=1
            CYTHON_TRACE_NOGIL=1
        )
    endif()
    
    target_include_directories(${TARGET_NAME} PRIVATE "${CYTHON_SOURCE_DIR}")
    
    install(TARGETS ${TARGET_NAME} DESTINATION stochmat)
endfunction()

# Define the modules to be built
add_cython_cpp_module(fast "src/stochmat/fast.pyx")
add_cython_cpp_module(_cython_sparse_stoch "src/stochmat/_cython_sparse_stoch.pyx")
