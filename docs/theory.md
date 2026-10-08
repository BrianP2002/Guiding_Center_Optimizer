# The method in one page

References: Ni & Wang, *Sensitivity analysis on chaotic dynamical systems by Non-Intrusive Least Squares Shadowing (NILSS)*,
arXiv:1611.00880 (equation numbers below are those of v9); Ni, Wang, Fernandez & Talnikar, *... by Finite Difference NILSS*,
arXiv:1711.06633 (the Schur-complement solve); Ni & Talnikar, *Adjoint sensitivity analysis ... by Non-Intrusive Least Squares Adjoint
Shadowing (NILSAS)*, arXiv:1801.08674 (the adjoint version, not implemented here).

## What is computed

A flow $\dot u = f(u; s)$ with a parameter $s$ (several in practice) and an objective $J(u)$ whose long-time average
$\langle J\rangle = \lim_{T\to\infty}\frac1T\int_0^T J(u(t))\,dt$ is to be differentiated: $d\langle J\rangle/ds$. In a chaotic
flow the tangent solutions grow exponentially, so the naive derivative of a long trajectory is useless (the butterfly effect), and
the derivative of an ensemble average by the tangent equations or an adjoint blows up with the time horizon.

**Shadowing.** For a uniformly hyperbolic attractor the shadowing lemma says that a perturbed trajectory exists that stays close to the
original one forever. Its first-order version is a tangent solution $v$ of

$$\dot v = \partial_u f\, v + \partial_s f ,$$

the *shadowing direction*, whose component perpendicular to the flow, $v^\perp = v - \frac{f^T v}{f^T f} f$ (eq. 8), is bounded.
The component along the flow is $\xi f$ with $\xi = f^T v / f^T f$: it is a time dilation. The sensitivity is the average of the
perturbation of $J$ along the shadowing direction plus a correction for the time dilation (eq. 41):

$$\frac{d\langle J\rangle}{ds} \approx \frac1T\sum_i\Big[\int_{t_i}^{t_{i+1}} \partial_u J\, v\,dt + \xi(t_{i+1})\big(\langle J\rangle - J(t_{i+1})\big)\Big]$$

(a term $\partial_s J$ is absent here: $J$ does not depend on the parameters).

**Least-squares shadowing (LSS)** approximates the bounded solution by the tangent solution with the smallest integrated perpendicular
norm, $\min_v \frac12\int_0^T\|v^\perp\|^2dt$ over all solutions of the tangent equation (eq. 16).

## NILSS: only the unstable subspace

All inhomogeneous tangent solutions differ by homogeneous ones, $\dot w = \partial_u f\,w$. Stable homogeneous solutions decay and
unstable ones grow, so only the unstable part of the difference has to be searched. Write

$$v = v^* + \sum_{j=1}^{n_{us}} a_j w_j ,$$

where $v^*$ is any inhomogeneous solution (started from zero), $w_j$ are $n_{us}$ homogeneous solutions started at random, and the
$a_j$ are the unknowns (eq. 17). The number $n_{us}$ (`nus`) must cover the positive Lyapunov exponents; the paper advises slightly
more. The cost is that of $n_{us}$ tangent solutions plus one per parameter, not that of the whole system.

**Segments and QR.** Both $v^*$ and the $w_j$ grow like $e^{\lambda t}$, so the trajectory is cut into $K$ segments of length `T_seg`.
At the end of segment $i$ the perpendicular homogeneous solutions are factorised, $W^\perp = Q R$, the new homogeneous tangents are the
columns of $Q$, and the part of $v^{*\perp}$ along $Q$ is moved into the coefficients: $b = Q^T v^{*\perp}$, $v^* \leftarrow v^{*\perp} - Q b$.
Continuity of $v^\perp$ across the interface is then a linear recursion for the coefficients (eq. 24),

$$a_{i+1} = R_i\, a_i + b_i .$$

The diagonal of $R$ gives the Lyapunov exponents of the projected tangent space, $\lambda_j \approx \frac{1}{K\Delta T}\sum_i\log|R_{i,jj}|$
(eq. 25); they are returned as `lyapunov`. The segment length should not exceed about $1/(\lambda_1-\lambda_{n_{us}})$ (sec. 4.2 of the
paper), otherwise the covariance matrices below become ill-conditioned.

**The least-squares problem.** With $C_i = \int (W_i^\perp)^T W_i^\perp\,dt$ (eq. 29) and $d_i = \int (W_i^\perp)^T v_i^{*\perp}\,dt$ (eq. 33),

$$\min_{a}\ \sum_i \Big(\tfrac12 a_i^T C_i a_i + d_i^T a_i\Big)\quad\text{s.t.}\quad a_{i+1} = R_i a_i + b_i$$

(eq. 37). Its Lagrange-multiplier (KKT) system is solved through the Schur complement $S = B\,C^{-1}B^T$ of the constraint matrix $B$,
which is block tridiagonal (arXiv:1711.06633); here it is assembled as a sparse matrix and factorised once
(`solve_shadowing_coeffs_multi`), at a cost linear in $K$.

## The streaming form

Everything the solution needs from the tangent solutions is a time integral over a segment: $C_i$, $d_i$, the integrals
$\int\partial_uJ\cdot v^*\,dt$ and $\int\partial_uJ\cdot w_j\,dt$ (the sensitivity is linear in $a$: $\int\partial_uJ\cdot v =
\int\partial_uJ\cdot v^* + \sum_j a_j\int\partial_uJ\cdot w_j$), and a few end-of-segment numbers ($R_i$, $b_i$, the ingredients of
$\xi$, $J(t_{i+1})$). These are accumulated step by step by the integrator (trapezoid rule, weight 1/2 at the interfaces), so the
solutions $u$, $W$, $v^*$ are never stored: the memory used for the tangents is that of the *current* segment, independent of the
number of steps and of $T$; what grows with $T$ is the list of small per-segment results (about $n_c\,(n_{us}+n_{par}) + n_{us}^2$ numbers
per segment, $n_c$ being the dimension of the state) and the sparse Schur system. A 2e5-time-unit run of the guiding-center flow therefore
needs the same memory as a 2e3 one (see [`hpc.md`](hpc.md)).
After the solve, the shadowing direction itself is rebuilt at the segment ends from $v^* + Wa$ to give the diagnostic `vnorm`.

## Several parameters at once

The homogeneous tangents do not depend on the parameter, so $W$, $R_i$, $C_i$, $B$ and $S$ are common to all parameters. Only $d_i$,
$b_i$ and $v^*$ differ: one inhomogeneous tangent per parameter is propagated with the same $W$, and the Schur system is factorised
once and solved for $n_{par}$ right-hand sides.

## Tangent equations by Jacobian-vector products

`rhs` is written once, with `jax.numpy`. At each Runge-Kutta stage `jax.linearize` gives the linear map $t\mapsto\partial_u f(u)\,t$,
which is applied (by `vmap`) to the $n_{us}$ rows of $W$ and the $n_{par}$ rows of $v^*$, and the parameter derivatives $\partial_s f$
are $n_{par}$ more forward-mode products: $n_{us}+2\,n_{par}$ Jacobian-vector products per stage, with no Jacobian matrix formed, so the
cost per step grows with the number of tangents and not with the dimension of the state. (The dense reference implementation,
`nilss_jax.reference`, forms the Jacobian by hand; the tests check that both give the same sensitivities to 1e-7.)

## Hamiltonian flows: the energy surface

For a Hamiltonian flow the energy $E(u;s)$ is conserved for every $s$. Then one more neutral direction exists (the energy), the
perturbed trajectory must lie on the perturbed surface $E(\cdot\,;s+\delta s) = E_0$, and the tangents must satisfy

$$\nabla_u E\cdot w = 0 ,\qquad \nabla_u E\cdot v^* + \partial_s E = 0 .$$

These conditions are preserved by the tangent equations: $E(\phi_t(u;s);s)$ does not depend on $t$, and differentiating it with respect to
$u$, respectively $s$, gives $\nabla E\cdot(\partial_u\phi_t\,w) = \nabla E(u)\cdot w$ and $\frac{d}{dt}\big(\nabla E\cdot v + \partial_sE\big) = 0$.
So it is enough to impose them at the start of the first segment, which `invariant=E` does: the random $w_j$ are projected onto
$\nabla E^\perp$ and $v^*$ is set to the minimum-norm solution $-\partial_sE\,\nabla E/\|\nabla E\|^2$ (both are then also made
perpendicular to $f$, which is consistent since $\nabla E\cdot f = 0$).

## Assumptions, and what happens without them

The derivation needs a uniformly hyperbolic attractor (so that the shadowing direction exists with a bounded perpendicular part) and an
ergodic one ($\langle J\rangle$ independent of the initial condition, so that the average along one orbit is the average of the system).
Lorenz 63 satisfies them well enough for the method to work as published. Systems with islands, mixed phase space or intermittency, such as
the guiding-center flow of a stellarator field with broken symmetry, do not: the shadowing direction then has a power-law tail, the
estimate from a run is dominated by rare events, and the error bar of the sample is meaningless. [`reliability.md`](reliability.md)
says what is checked and how, and [`findings.md`](findings.md) reports the measurements.
