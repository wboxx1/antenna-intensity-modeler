from antenna_intensity_modeler import parabolic
import matplotlib.pyplot as plt
params = parabolic.parameters(2.4, 8.4e9, 400.0, 0.62, 35)
params['ffpwrden'] = 1.
xbar = 0.
table = parabolic.near_field_corrections(params, xbar)
fig, ax = plt.subplots()
ax.semilogx(table.delta, table.Pcorr)
ax.set_xlim([0.01, 1.0])
ax.grid(True, which="both")
ax.minorticks_on()
side_lobe_ratio = params['side_lobe_ratio']
ax.set_title("Near Field Corrections xbar: %s , slr: %s" % (xbar, side_lobe_ratio))
ax.set_xlabel("Normalized On Axis Distance")
ax.set_ylabel("Normalized On Axis Power Density")
plt.show()