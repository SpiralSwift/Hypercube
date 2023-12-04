# Hypercube
A simple hypercube renderer

Creates a tesseract by default;
other objects can be used either by entering values in the CLI (or by changing default parameters in the script).

The script takes two optional command-line inputs:
1. (int) number of dimensions (minimum 2)
2. (int) "step" value for determining which pairs of axes are used to rotate (ex: step = 2 --> axes 0 & 2, 1 & 3, etc.)

Setting the number of dimensions > 10 is not recommended, as this will run rather slowly past that point.
The "step" value should probably be on [1,3]; higher step values are better for higher dimensions.

Line colors indicate which axis a given edge runs along.
