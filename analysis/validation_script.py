

# Generate data
# z ~ p(z), z dep x, y
# x and y generate z
# It should be possible to recover them independently
# The same reasoning motivates the EEG case

def f_z(x, y):
	return z

x = torch.rand(10)
y = torch.rand(10)
z = f_z(x, y)

data = Dataset(x, y, z)
