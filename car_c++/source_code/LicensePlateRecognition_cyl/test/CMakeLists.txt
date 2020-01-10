add_executable( LicensePlate_Recognition Main.cpp )
target_link_libraries( LicensePlate_Recognition mylib )

add_executable( DatasetForTrain GenData.cpp )
target_link_libraries( DatasetForTrain mylib )

