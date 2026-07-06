FILE(GLOB FILES "${HIP_SOURCE_DIR}/kernels/*.hip")
SET(OUTPUT_CPP "${HIP_KERNELS_CPP}")
SET(OUTPUT_H "${HIP_KERNELS_H}")

FILE(WRITE "${OUTPUT_CPP}" "#include \"${HIP_SOURCE_CLASS}.h\"\n#include <string>\nusing namespace std;\n")
FILE(WRITE "${OUTPUT_H}" "#ifndef ${HIP_SOURCE_CLASS}_H_\n#define ${HIP_SOURCE_CLASS}_H_\n#include <string>\nclass ${HIP_SOURCE_CLASS} {\npublic:\n")

FOREACH(FILE_NAME ${FILES})
    GET_FILENAME_COMPONENT(FILE_BASE ${FILE_NAME} NAME_WE)
    FILE(READ ${FILE_NAME} FILE_CONTENT)
    STRING(REGEX REPLACE "\\\\" "\\\\\\\\" FILE_CONTENT "${FILE_CONTENT}")
    STRING(REGEX REPLACE "\"" "\\\\\"" FILE_CONTENT "${FILE_CONTENT}")
    STRING(REGEX REPLACE "\n" "\\\\n\"\n\"" FILE_CONTENT "${FILE_CONTENT}")
    FILE(APPEND "${OUTPUT_CPP}" "const string ${HIP_SOURCE_CLASS}::${FILE_BASE} = \"${FILE_CONTENT}\";\n")
    FILE(APPEND "${OUTPUT_H}" "static const std::string ${FILE_BASE};\n")
ENDFOREACH()

FILE(APPEND "${OUTPUT_H}" "};\n#endif // ${HIP_SOURCE_CLASS}_H_\n")
