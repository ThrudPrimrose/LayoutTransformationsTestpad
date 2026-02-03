g++ -march=sapphirerapids -O3 -o amx_gemm_exe amx_gemm.cpp
sde64 -spr -- ./amx_gemm_exe
