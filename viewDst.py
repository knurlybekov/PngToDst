import matplotlib.pyplot as plt
from pyembroidery import read_dst


pattern = read_dst('output.dst')  # DST file

# Extract stitches
stitches = pattern.stitches

# Separate X and Y coordinates
x_coords = [point[0] for point in stitches]
y_coords = [point[1] for point in stitches]

# Plot the pattern
plt.figure(figsize=(8, 8))
plt.plot(x_coords, y_coords, 'b-', linewidth=0.75)
plt.gca().invert_yaxis()  # Invert Y-axis to match the typical DST coordinate system
plt.title('DST Stitch Pattern Visualization')
plt.axis('equal')
plt.show()
