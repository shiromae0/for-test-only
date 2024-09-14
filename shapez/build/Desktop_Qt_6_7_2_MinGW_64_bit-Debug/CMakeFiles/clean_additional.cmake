# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "Debug")
  file(REMOVE_RECURSE
<<<<<<< HEAD
  "CMakeFiles\\MyShapez_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\MyShapez_autogen.dir\\ParseCache.txt"
=======
  "CMakeFiles\\MyShapezLib_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\MyShapezLib_autogen.dir\\ParseCache.txt"
  "CMakeFiles\\MyShapez_autogen.dir\\AutogenUsed.txt"
  "CMakeFiles\\MyShapez_autogen.dir\\ParseCache.txt"
  "MyShapezLib_autogen"
>>>>>>> 8fcbed69569fc321d69e664afc48e50f066e3ae1
  "MyShapez_autogen"
  )
endif()
