import matplotlib.pyplot as plt
#target_file = 'fid_stats.txt'
target_file = 'fid_stats_wgan-gp.txt'
with open(target_file) as f:
	lines = [line.rstrip('\n') for line in open(target_file)]
epochs = []
fid = []
for line in lines:
	try:
		epoch_str, fid_str = line.split()
		epochs.append(int(epoch_str))
		fid.append(float(fid_str))
	except:
		print('done')
print(fid)
plt.ylabel('Fréchet Inception Distance')
plt.xlabel('epoch\n wgan-gp')
plt.plot(epochs, fid)
#plt.axis([300, 600, 300, 600])
plt.show()
#for
#read one line 
#append fid
#append epoch
#plot
