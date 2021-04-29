set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DELPP_NO_DEFAULT_LOG_FILE")

#file(GLOB_RECURSE SRC *)
file(GLOB SRC *.cc *.cu)
cuda_add_library(${PROJECT_LIB_NAME} ${SRC} OPTIONS -DYAML_BUILD_SHARED_LIBS=ON ) #OPTIONS -shared -fPIC
# CUDA_ADD_EXECUTABLE(${PROJECT_LIB_NAME} )
target_link_libraries(${PROJECT_LIB_NAME} ${LINK_LIBRARY} OpenMP::OpenMP_CXX)

# Export the package for use from the build tree
# export(TARGETS ${PROJECT_NAME} NAMESPACE XComp:: FILE cmake/${PROJECT_NAME}Targets.cmake)
# export(PACKAGE ${PROJECT_NAME})

# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
#     cmake/${PROJECT_NAME}ConfigVersion.cmake
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY AnyNewerVersion)

# configure_package_config_file(
#     ${CMAKE_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
#     ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake
#     INSTALL_DESTINATION lib/cmake/)

# install(TARGETS ${PROJECT_NAME}-train ${PROJECT_NAME}-predict
#     RUNTIME DESTINATION bin
#     BUNDLE DESTINATION bin)

# install(TARGETS ${PROJECT_NAME}
#     EXPORT ${PROJECT_NAME}Targets
#     ARCHIVE DESTINATION lib
#     LIBRARY DESTINATION lib
#     RUNTIME DESTINATION bin
#     INCLUDES DESTINATION include)
# install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/thundersvm DESTINATION include/)
# install(FILES ${CMAKE_BINARY_DIR}/thundersvm/config.h DESTINATION include/thundersvm/)

# install(FILES
#     ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake
#     ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake
#     DESTINATION lib/cmake/)

# install(EXPORT ${PROJECT_NAME}Targets
#     NAMESPACE XComp:: DESTINATION lib/cmake/)