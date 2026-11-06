include(GoogleTest)

# Tests using Google Test
set(GTEST_TARGETS
    test_gemm
    test_greenstrain
    test_hadamard
    test_isotropic
    test_matdet
    test_matsum
    test_matinv
    test_mattrace
    test_matvecmult
    test_quaternion
    test_symeigs
    test_matmulttrace
    test_symrk
    test_symsum
    test_veccross
    test_vecnorm
    test_vecouter
    test_vecsum
    test_a2dmat
    test_a2dmatinv
    test_a2dmatdet
    test_ad_expressions
    test_bspline
    test_adscalar
)

foreach(target IN LISTS GTEST_TARGETS)
    add_executable(${target} ${target}.cpp)

    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/tests
    )

    target_link_libraries(${target} PRIVATE gtest_main)

    gtest_discover_tests(${target})
endforeach()

add_subdirectory(core)
