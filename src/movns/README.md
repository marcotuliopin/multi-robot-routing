### Neighborhoods:

- **k = 0**
    - Local search = *move_point*
    - Perturb solution = *two_opt*
- **k = 1**
    - Local search = *swap_points*
    - Perturb solution = *two_opt*
- **k = 2**
    - Local search = *swap_points* in both paths
    - PS = *two_opt*
- **k = 3**
    - LS = *move_point*
    - PS = *swap_subpaths*
- **k = 4**
    - LS = *swap_points*
    - PS = *swap_subpaths*
- **k = 5**
    - LS = *swap_points* in both paths
    - PS = *swap_subpaths*
