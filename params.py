"""Parameters shared by Stefan_notebook.ipynb and Stefan_notebook_forward_euler.ipynb."""

# Material data (SI units)
rho_L = 999.8     # density [kg/m^3]
c_L   = 4210.0    # specific heat [J/(kg K)]
k_L   = 0.567     # thermal conductivity [W/(m K)]
L_f   = 333.4e3   # latent heat [J/kg]
m_L   = 10.0      # temperature at x = 0 [K above melting]

a_L = k_L / (rho_L * c_L)   # diffusivity [m^2/s]
mu  = L_f / c_L             # [K]
St  = c_L * m_L / L_f       # Stefan number

# Test cases
T     = 3600.0              # final time [s], comparison with the analytical solution
s0    = 0.1                 # initial front position [m], temporal EOC test
T_eoc = 0.1 * s0**2 / a_L   # final time [s], temporal EOC test
