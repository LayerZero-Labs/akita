#if defined(__APPLE__)
#define C_SYMBOL(name) _##name
#else
#define C_SYMBOL(name) name
#endif

.text
.p2align 2
.globl C_SYMBOL(akita_fp128_mul_asm)
#if !defined(__APPLE__)
.type C_SYMBOL(akita_fp128_mul_asm), %function
#endif
C_SYMBOL(akita_fp128_mul_asm):
#include "fp128_mul_body.inc"
ret
#if !defined(__APPLE__)
.size C_SYMBOL(akita_fp128_mul_asm), .-C_SYMBOL(akita_fp128_mul_asm)
.section .note.GNU-stack,"",%progbits
#endif
