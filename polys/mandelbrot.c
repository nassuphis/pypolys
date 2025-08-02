/* mandelbrot.c
 *
 * Build the degree-255 Mandelbrot polynomial (p_{i+1}=1+x p_i^2),
 * solve it with MPSolve’s monomial API, and print all 255 roots.
 */

#include <mps/mps.h>
#include <mps/monomial-poly.h>
#include <mps/interface.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int k      = 8;
    const int degree = (1 << k) - 1;    // 255
    const int N      = degree + 1;      // 256 coefficients

    /* 1) Build coefficients in double[] */
    double *coeffs = calloc(N, sizeof(double));
    if (!coeffs) { perror("calloc"); return EXIT_FAILURE; }
    coeffs[0] = 1.0;
    for (int iter = 1; iter <= k; ++iter) {
        int prev = (1 << (iter - 1)) - 1;
        int now  = (1 << iter)       - 1;
        double *tmp = calloc(now + 1, sizeof(double));
        if (!tmp) { perror("calloc"); free(coeffs); return EXIT_FAILURE; }
        for (int i = 0; i <= prev; ++i)
            for (int j = 0; j <= prev; ++j)
                tmp[i + j + 1] += coeffs[i] * coeffs[j];
        tmp[0] += 1.0;
        free(coeffs);
        coeffs = tmp;
    }

    /* 2) Create and configure MPSolve context */
    mps_context *ctx = mps_context_new();
    if (!ctx) {
        fprintf(stderr, "Failed to create MPS context\n");
        free(coeffs);
        return EXIT_FAILURE;
    }
    mps_context_set_input_prec(ctx, 512);
    mps_context_set_output_prec(ctx, 512);
    mps_context_set_output_goal(ctx, MPS_OUTPUT_GOAL_APPROXIMATE);

    /* 3) Load coefficients into a monomial-poly and hand off */
    mps_monomial_poly *mp = mps_monomial_poly_new(ctx, degree);
    if (!mp) {
        fprintf(stderr, "Failed to create monomial-poly\n");
        mps_context_free(ctx);
        free(coeffs);
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        mps_monomial_poly_set_coefficient_d(ctx, mp, i, coeffs[i], 0.0);
    }
    mps_context_set_input_poly(ctx, MPS_POLYNOMIAL(mp));

    /* 4) Solve (void return) */
    mps_mpsolve(ctx);

    /* 5) Retrieve roots (void) and print exactly `degree` of them */
    cplx_t *roots = NULL;
    mps_context_get_roots_d(ctx, &roots, NULL);
    for (int i = 0; i < degree; ++i) {
        printf("%.15g + %.15gi\n",
               cplx_Re(roots[i]), cplx_Im(roots[i]));
    }
    cplx_vfree(roots);

    /* 6) Cleanup */
    mps_polynomial_free(ctx, MPS_POLYNOMIAL(mp));
    mps_context_free(ctx);
    free(coeffs);

    return EXIT_SUCCESS;
}
