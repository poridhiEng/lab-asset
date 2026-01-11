cmake_minimum_required(VERSION 3.18)
project(lab34 LANGUAGES CUDA CXX)

# Stream scaling experiment
add_executable(stream_scaling stream_scaling.cu)
target_compile_options(stream_scaling
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream legacy>
)

# SM saturation experiment
add_executable(sm_saturation sm_saturation.cu)
target_compile_options(sm_saturation
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--default-stream legacy>
)
