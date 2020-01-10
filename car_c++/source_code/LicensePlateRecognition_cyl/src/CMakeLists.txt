add_library( mylib SHARED
    DetectChars.cpp
    DetectPlates.cpp
    PossibleChar.cpp
    PossiblePlate.cpp
    Preprocess.cpp
)

target_link_libraries( mylib
    ${THIRD_PARTY_LIBS}
)
