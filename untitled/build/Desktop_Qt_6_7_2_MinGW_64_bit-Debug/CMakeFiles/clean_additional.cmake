# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
  "CMakeFiles\\untitled_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\untitled_autogen.dir\\ParseCache.txt"
  "CMakeFiles\\untitledlib_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\untitledlib_autogen.dir\\ParseCache.txt"
  "untitled_autogen"
  "untitledlib_autogen"
  )
endif()
