import mpmath as mp
def bozo_aberth(coeffs, maxsteps, roots_init):
    n = len(coeffs) - 1
    cf = [c/coeffs[0] for c in coeffs]
    roots = [roots_init**k for k in range(n)]
    for _ in range(maxsteps):
        for i in range(n):
            pi = roots[i]
            delta = mp.mp.polyval(cf, pi)
            for j in range(n):
                if i == j: continue
                try:
                    delta /= (pi - roots[j])
                except ZeroDivisionError:
                    pass
            roots[i] = pi - delta
    return roots
   
 