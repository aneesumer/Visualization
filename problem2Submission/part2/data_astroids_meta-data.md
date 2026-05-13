The file sbdb.csv is a standard CSV file. It contains information about small bodies in the solar system, obtained from the JPL/NASA Small-Body Database via
https://ssd.jpl.nasa.gov/tools/sbdb_query.html
with some minimal processing.
It may help to look up a few technical terms on wikipedia to understand this problem.
Each row describes one object. The columns of the file are:
full_name, string, name of the oject
a, float, semi-major axis of the approximate elliptical orbit of the object in astronomical units (AU)
e, float, eccentricity of the approximate elliptical orbit of the object
i, float, orbital inclination in degrees
H, float, absolute magnitude of the object, inverse logarithmic scale, lower means brighter (under some "standard astronomical conditions")
diameter, float, approximate diameter of the object in kilometers
albedo, float, albedo of the object
class, categorical, type of asteroid (see below for list of classes)
first_obs, date, day of first observation of the object

classes (short and full name):
AMO : Amor
APO : Apollo
AST : Asteroid (other)
ATE : Aten
CEN : Centaur
HYA : Hyperbolic Asteroid
IEO : Atira
IMB : Inner Main-belt Asteroid
MBA : Main-belt Asteroid
MCA : Mars Crossing Asteroid
OMB : Outer Main-belt Asteroid
TJN : Jupiter Trojan
TNO : TransNeptunian Object

