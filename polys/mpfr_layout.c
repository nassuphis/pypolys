/* mpfr_layout.c  (layout‑dump 1.0) */
#include <stdio.h>
#include <stddef.h>
#include <mpfr.h>

#define SHOW(field) \
    printf("  %-4s offset=%2zu  size=%zu\n", #field,                \
           offsetof(__mpfr_struct, field),                          \
           sizeof(((mpfr_ptr)0)->field))

int main(void)
{
    printf("sizeof(mpfr_t)=%zu\n", sizeof(mpfr_t));
    SHOW(_mpfr_prec);
    SHOW(_mpfr_sign);
    SHOW(_mpfr_exp);
    SHOW(_mpfr_d);
    return 0;
}
