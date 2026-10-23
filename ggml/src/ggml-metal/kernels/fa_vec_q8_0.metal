#include "common.h"
#include "dequantize.h"
#include "fa_vec_common.metal"

#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk32_dv32_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 32, 32, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q1_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q2_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q4_ne2")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk64_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 64, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv64")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 64, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv64_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 64, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk96_dv64_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 96, 64, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv96")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv96_q2_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 96, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk128_dv96_q4_ne4")]]   kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 96, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv192_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk192_dv128_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk256_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk320_dv256_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q1_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne1")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk512_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 4, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q1_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 1>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q2_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q2_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 2>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q4_ne2")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2, 4>;
template [[host_name("kernel_flash_attn_ext_vec_q8_0_dk576_dv512_q4_ne4")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 4, 4>;




template<
    typename q4_t,  // query types in shared memory
    typename k4_t,  // key types in shared memory
    typename v4_t,  // value types in shared memory
    typename qk_t,  // Q*K types
    typename s_t,   // soft-max types
    typename s4_t,
    typename o4_t,  // attention accumulation types
    typename kd4_t, // key type in device memory
    short nl_k,
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t, // value type in device memory
    short nl_v,
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,       // K head size
    short DV,       // V head size
    short NE = 4,   // head elements per thread
    short H  = 1,   // query heads per threadgroup
    short Q  = OP_FLASH_ATTN_EXT_VEC_NQPSG,  // queries per threadgroup
    short C  = OP_FLASH_ATTN_EXT_VEC_NCPSG>  // cache items per threadgroup
kernel void kernel_flash_attn_ext_vec_gqa2(
        constant ggml_metal_kargs_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    static_assert(DK % 32 == 0, "DK must be divisible by 32");
    static_assert(DV % 32 == 0, "DV must be divisible by 32");
    static_assert(H == 1 || H == 2, "unsupported query heads per threadgroup");

#define NWG  (FC_flash_attn_ext_vec_nwg)
#define NSG  (FC_flash_attn_ext_vec_nsg)

#define NS10 (FC_flash_attn_ext_vec_ns10)
#define NS20 (FC_flash_attn_ext_vec_ns20)

    const short iwg = tgpig[2]%NWG;

    const ushort iq3 = tgpig[2]/NWG;
    const ushort iq2 = H*tgpig[1];
    const ushort iq1 = tgpig[0];

    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;

    constexpr short PK  = PAD2(DK, 128);
    constexpr short PK4 = PK/4;

    constexpr short PV  = PAD2(DV, 128);
    constexpr short PV4 = PV/4;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NL  = NW/NE; // note: this can be adjusted to support different head sizes and simdgroup work loads
    constexpr short SH  = 4*C;   // shared memory per simdgroup

    static_assert(DK4 % NL == 0, "DK4 must be divisible by NL");
    static_assert(DV4 % NL == 0, "DV4 must be divisible by NL");

    threadgroup q4_t * sq4 = (threadgroup q4_t *) shmem_f16;
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + NSG*H*PK);
    threadgroup s4_t * ss4 = (threadgroup s4_t *) (shmem_f16 + NSG*H*PK);
    threadgroup half * sm  = (threadgroup half *) (shmem_f16 + NSG*H*PK);
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + NSG*H*(PK + SH));

    {
        q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

        const short ikv2 = iq2/(args.ne02/args.ne_12_2);
        const short ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    if (iq1 < args.ne01) {
        FOR_UNROLL (short ih = 0; ih < H; ++ih) {
            device const float4 * q4 = (device const float4 *) (q + ih*args.nb02);

            for (short i = tiisg; i < PK4; i += NW) {
                if (i < DK4) {
                    sq4[ih*PK4 + i] = (q4_t) q4[i];
                } else {
                    sq4[ih*PK4 + i] = (q4_t) 0.0f;
                }
            }
        }
    }

    // zero out so
    FOR_UNROLL (short ih = 0; ih < H; ++ih) {
        threadgroup o4_t * so4h = so4 + (sgitg*H + ih)*PV4 + tiisg;

        for (short i = 0; i < DV4/NL; ++i) {
            so4h[i*NL] = (o4_t) 0.0f;
        }
    }

    // zero out shared memory SH
    for (short i = tiisg; i < H*SH/4; i += NW) {
        ss4[sgitg*H*SH/4 + i] = (s4_t) 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        float S[H];
        float M[H];

        FOR_UNROLL (short ih = 0; ih < H; ++ih) {
            S[ih] = 0.0f;
            M[ih] = -FLT_MAX/2;
        }

        // thread indices inside the simdgroup
        const short tx = tiisg%NL;
        const short ty = tiisg/NL;

        device const half * pm = (device const half *) (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
        threadgroup half * sm0 = sm + sgitg*H*SH + 2*C;

        // loop over the KV cache
        // each simdgroup handles blocks of Q rows and C columns
        for (int ic0 = iwg*NSG + sgitg; ; ic0 += NWG*NSG) {
            int ic = ic0*C;
            if (ic >= args.ne11) {
                break;
            }

            // the last partial chunk uses the pad buffer as source
            if (FC_flash_attn_ext_vec_has_kvpad && ic + C > args.ne11) {
                k    = pad;
                v    = k + args.nb11*C*args.ne_12_2*args.ne_12_3;
                mask = v + args.nb21*C*args.ne_12_2*args.ne_12_3;

                const short ikv2 = iq2/(args.ne02/args.ne_12_2);
                const short ikv3 = iq3/(args.ne03/args.ne_12_3);

                k += (ikv2 + ikv3*args.ne_12_2)*args.nb11*C;
                v += (ikv2 + ikv3*args.ne_12_2)*args.nb21*C;

                if (!FC_flash_attn_ext_vec_has_mask) {
                    if (ic + tiisg >= args.ne11) {
                        sm0[tiisg] = -MAXHALF;
                    }
                } else {
                    pm = (device const half *) (mask) +
                        iq1*C +
                        (iq2%args.ne32)*(C*args.ne31) +
                        (iq3%args.ne33)*(C*args.ne31*args.ne32);
                }

                ic = 0;
            }

            if (FC_flash_attn_ext_vec_has_mask) {
                sm0[tiisg] = pm[ic + tiisg];
            }

            // skip -INF blocks
            if (simd_max(sm0[tiisg]) <= -MAXHALF) {
                continue;
            }

            // Q*K^T
            {
                device const k4_t * pk4 = (device const k4_t *) (k + ic*args.nb11);

                pk4 += ty*NS10/4 + tx;

                qk_t mqk[H][C/NE];

                FOR_UNROLL (short ih = 0; ih < H; ++ih) {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        mqk[ih][cc] = 0.0f;
                    }
                }

                // each simdgroup processes H query heads and NE (NW/NL) cache elements
                FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                    if (is_same<kd4_t, k4_t>::value) {
                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const k4_t mk = pk4[cc*NE*NS10/4 + ii*NL];
                            const short i = ii*NL + tx;

                            FOR_UNROLL (short ih = 0; ih < H; ++ih) {
                                mqk[ih][cc] += dot((float4) mk, (float4) sq4[ih*PK4 + i]);
                            }
                        }
                    } else {
                        device const kd4_t * pk = (device const kd4_t *) (k + ((ic + NE*cc + ty)*args.nb11));

                        k4_t mk;

                        FOR_UNROLL (short ii = 0; ii < DK4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            deq_k_t4(pk + i/nl_k, i%nl_k, mk);

                            FOR_UNROLL (short ih = 0; ih < H; ++ih) {
                                mqk[ih][cc] += dot((float4) mk, (float4) sq4[ih*PK4 + i]);
                            }
                        }
                    }

                    FOR_UNROLL (short ih = 0; ih < H; ++ih) {
                        if (NE == 1) {
                            mqk[ih][cc] = simd_sum(mqk[ih][cc]);
                        } else {
                            if (NE <= 1) {
                                mqk[ih][cc] += simd_shuffle_down(mqk[ih][cc], 16);
                            }
                            if (NE <= 2) {
                                mqk[ih][cc] += simd_shuffle_down(mqk[ih][cc],  8);
                            }
                            if (NE <= 4) {
                                mqk[ih][cc] += simd_shuffle_down(mqk[ih][cc],  4);
                            }
                            if (NE <= 8) {
                                mqk[ih][cc] += simd_shuffle_down(mqk[ih][cc],  2);
                            }
                            if (NE <= 16) {
                                mqk[ih][cc] += simd_shuffle_down(mqk[ih][cc],  1);
                            }

                            mqk[ih][cc] = simd_shuffle(mqk[ih][cc], NL*ty);
                        }
                    }
                }

                FOR_UNROLL (short ih = 0; ih < H; ++ih) {
                    threadgroup s_t * ssh = ss + (sgitg*H + ih)*(SH/2);

                    if (FC_flash_attn_ext_vec_has_mask &&
                       !FC_flash_attn_ext_vec_has_scap &&
                       !FC_flash_attn_ext_vec_has_bias) {
                        ssh[NE*tx + ty] = fma(mqk[ih][tx], args.scale, (qk_t) sm0[NE*tx + ty]);
                    } else {
                        mqk[ih][tx] *= args.scale;

                        if (FC_flash_attn_ext_vec_has_scap) {
                            mqk[ih][tx] = args.logit_softcap*precise::tanh(mqk[ih][tx]);
                        }

                        if (FC_flash_attn_ext_vec_has_bias) {
                            const short h = iq2 + ih;
                            const float base = h < args.n_head_log2 ? args.m0 : args.m1;
                            const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;
                            const float slope = pow(base, exph);

                            mqk[ih][tx] += (qk_t) sm0[NE*tx + ty]*slope;
                        } else {
                            mqk[ih][tx] += (qk_t) sm0[NE*tx + ty];
                        }

                        ssh[NE*tx + ty] = mqk[ih][tx];
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // online softmax
            {
                FOR_UNROLL (short ih = 0; ih < H; ++ih) {
                    threadgroup s_t  * ssh  = ss  + (sgitg*H + ih)*(SH/2);
                    threadgroup o4_t * so4h = so4 + (sgitg*H + ih)*PV4 + tiisg;

                    const float m = M[ih];
                    const float s = ssh[tiisg];

                    M[ih] = simd_max(max(M[ih], s));

                    const float ms = exp(m - M[ih]);
                    const float vs = exp(s - M[ih]);

                    S[ih] = S[ih]*ms + simd_sum(vs);

                    ssh[tiisg] = vs;

                    if ((DV4/NL % NW == 0) || ty == 0) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            so4h[ii*NL] *= ms;
                        }
                    }
                }
            }

            simdgroup_barrier(mem_flags::mem_threadgroup);

            // O = O + (Q*K^T)*V
            {
                o4_t lo[H][DV4/NL];
                FOR_UNROLL (short ih = 0; ih < H; ++ih) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        lo[ih][ii] = 0.0f;
                    }
                }

                if (is_same<vd4_t, v4_t>::value) {
                    device const v4_t * pv4 = (device const v4_t *) (v + ic*args.nb21);

                    pv4 += ty*NS20/4 + tx;

                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const v4_t mv = pv4[cc*NE*NS20/4 + ii*NL];

                            FOR_UNROLL (short ih = 0; ih < H; ++ih) {
                                threadgroup s_t * ssh = ss + (sgitg*H + ih)*(SH/2);
                                lo[ih][ii] += o4_t(float4(mv)*float4(ssh[NE*cc + ty]));
                            }
                        }
                    }
                } else {
                    FOR_UNROLL (short cc = 0; cc < C/NE; ++cc) {
                        device const vd4_t * pv4 = (device const vd4_t *) (v + ((ic + NE*cc + ty)*args.nb21));

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            const short i = ii*NL + tx;

                            v4_t mv;
                            deq_v_t4(pv4 + i/nl_v, i%nl_v, mv);

                            FOR_UNROLL (short ih = 0; ih < H; ++ih) {
                                threadgroup s_t * ssh = ss + (sgitg*H + ih)*(SH/2);
                                lo[ih][ii] += o4_t(float4(mv)*float4(ssh[NE*cc + ty]));
                            }
                        }
                    }
                }

                FOR_UNROLL (short ih = 0; ih < H; ++ih) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        if (NE > 1) {
                            lo[ih][ii][0] += simd_shuffle_down(lo[ih][ii][0], 16);
                            lo[ih][ii][1] += simd_shuffle_down(lo[ih][ii][1], 16);
                            lo[ih][ii][2] += simd_shuffle_down(lo[ih][ii][2], 16);
                            lo[ih][ii][3] += simd_shuffle_down(lo[ih][ii][3], 16);
                        }

                        if (NE > 2) {
                            lo[ih][ii][0] += simd_shuffle_down(lo[ih][ii][0],  8);
                            lo[ih][ii][1] += simd_shuffle_down(lo[ih][ii][1],  8);
                            lo[ih][ii][2] += simd_shuffle_down(lo[ih][ii][2],  8);
                            lo[ih][ii][3] += simd_shuffle_down(lo[ih][ii][3],  8);
                        }

                        if (NE > 4) {
                            lo[ih][ii][0] += simd_shuffle_down(lo[ih][ii][0],  4);
                            lo[ih][ii][1] += simd_shuffle_down(lo[ih][ii][1],  4);
                            lo[ih][ii][2] += simd_shuffle_down(lo[ih][ii][2],  4);
                            lo[ih][ii][3] += simd_shuffle_down(lo[ih][ii][3],  4);
                        }

                        if (NE > 8) {
                            lo[ih][ii][0] += simd_shuffle_down(lo[ih][ii][0],  2);
                            lo[ih][ii][1] += simd_shuffle_down(lo[ih][ii][1],  2);
                            lo[ih][ii][2] += simd_shuffle_down(lo[ih][ii][2],  2);
                            lo[ih][ii][3] += simd_shuffle_down(lo[ih][ii][3],  2);
                        }

                        if (NE > 16) {
                            lo[ih][ii][0] += simd_shuffle_down(lo[ih][ii][0],  1);
                            lo[ih][ii][1] += simd_shuffle_down(lo[ih][ii][1],  1);
                            lo[ih][ii][2] += simd_shuffle_down(lo[ih][ii][2],  1);
                            lo[ih][ii][3] += simd_shuffle_down(lo[ih][ii][3],  1);
                        }
                    }

                    if ((DV4/NL % NW == 0) || ty == 0) {
                        threadgroup o4_t * so4h = so4 + (sgitg*H + ih)*PV4 + tiisg;

                        FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                            so4h[ii*NL] += lo[ih][ii];
                        }
                    }
                }
            }
        }

        if (FC_flash_attn_ext_vec_has_sinks && sgitg == 0 && iwg == 0) {
            FOR_UNROLL (short ih = 0; ih < H; ++ih) {
                threadgroup o4_t * so4h = so4 + ih*PV4 + tiisg;

                const float m = M[ih];
                const float s = tiisg == 0 ? ((device const float *) sinks)[iq2 + ih] : -FLT_MAX/2;

                M[ih] = simd_max(max(M[ih], s));

                const float ms = exp(m - M[ih]);
                const float vs = exp(s - M[ih]);

                S[ih] = S[ih]*ms + simd_sum(vs);

                if ((DV4/NL % NW == 0) || ty == 0) {
                    FOR_UNROLL (short ii = 0; ii < DV4/NL; ++ii) {
                        so4h[ii*NL] *= ms;
                    }
                }
            }
        }

        // these are needed for reducing the results from the simdgroups (reuse the ss buffer)
        if (tiisg == 0) {
            FOR_UNROLL (short ih = 0; ih < H; ++ih) {
                threadgroup s_t * ssh = ss + (sgitg*H + ih)*(SH/2);
                ssh[0] = (s_t) S[ih];
                ssh[1] = (s_t) M[ih];
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel reduce
    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            FOR_UNROLL (short ih = 0; ih < H; ++ih) {
                threadgroup s_t * ss0 = ss + (sgitg*H + ih)*(SH/2);
                threadgroup s_t * ss1 = ss + ((sgitg + r)*H + ih)*(SH/2);

                const float S0 = ss0[0];
                const float S1 = ss1[0];

                const float M0 = ss0[1];
                const float M1 = ss1[1];

                const float M = max(M0, M1);

                const float ms0 = exp(M0 - M);
                const float ms1 = exp(M1 - M);

                const float S = S0*ms0 + S1*ms1;

                if (tiisg == 0) {
                    ss0[0] = S;
                    ss0[1] = M;
                }

                threadgroup o4_t * so0 = so4 + (sgitg*H + ih)*PV4;
                threadgroup o4_t * so1 = so4 + ((sgitg + r)*H + ih)*PV4;

                for (short i = tiisg; i < DV4; i += NW) {
                    so0[i] = so0[i]*ms0 + so1[i]*ms1;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // final rescale with 1/S and store to global memory
    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;

        device float4 * dst4 = (device float4 *) dst;
        device float  * dst1 = (device float  *) dst + nrows*DV*NWG; // the S and M are stored after the results

        FOR_UNROLL (short ih = 0; ih < H; ++ih) {
            const int64_t rid = iq3*args.ne2*args.ne1 + iq2 + ih + iq1*args.ne1;
            threadgroup s_t  * ssh  = ss  + ih*(SH/2);
            threadgroup o4_t * so4h = so4 + ih*PV4;

            const float scale = NWG == 1 ? (ssh[0] == 0.0f ? 0.0f : 1.0f/ssh[0]) : 1.0f;

            for (short i = tiisg; i < DV4; i += NW) {
                dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4h[i]*scale;
            }

            if (NWG > 1) {
                if (tiisg == 0) {
                    dst1[rid*(2*NWG) + 2*iwg + 0] = ssh[0];
                    dst1[rid*(2*NWG) + 2*iwg + 1] = ssh[1];
                }
            }
        }
    }

#undef NWG
#undef NSG
#undef NS10
#undef NS20
}

// note: I think the s_t can be half instead of float, because the Q*K scaling is done before storing to shared mem
//       in the other (non-vec) kernel, we need s_t to also be float because we scale during the soft_max
//
#define FA_TYPES \
           half4,  \
           half4,  \
           half4,  \
    float,         \
    float, float4, \
           float4

#define FA_TYPES_F32 \
           half4,  \
           float4, \
           float4, \
    float,         \
    float, float4, \
           float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4>) flash_attn_ext_vec_t;

typedef decltype(kernel_flash_attn_ext_vec_gqa2<FA_TYPES, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4, 128, 128, 4, 1, 1>) flash_attn_ext_vec_gqa2_t;
template [[host_name("kernel_flash_attn_ext_vec_gqa2_q8_0_dk256_dv256")]] kernel flash_attn_ext_vec_gqa2_t kernel_flash_attn_ext_vec_gqa2<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4_packed, block_q8_0, 8, dequantize_q8_0_t4_packed, 256, 256, 1, 2, 1>;

#undef FA_TYPES
#undef FA_TYPES_F32
