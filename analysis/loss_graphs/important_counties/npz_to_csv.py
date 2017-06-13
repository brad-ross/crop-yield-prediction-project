import numpy
import sys
import os

def convert_to_csv(filenames,outputname):
	l = numpy.zeros((125,1))
	print(l.shape)
	for filename in filenames:
		file = numpy.load(filename)
		print(file['summary_train_loss'].shape)
		#l = numpy.concatenate((l, file['summary_train_loss'][:, numpy.newaxis], file['summary_eval_loss'][:, numpy.newaxis]),axis=1)
		l = numpy.concatenate((l, file['summary_RMSE'][:, numpy.newaxis]),axis=1)
	numpy.savetxt(outputname, l, delimiter=",")

filenames = ['run0__dropout_0.25__soybean.npz', 'run1__dropout_0.50__soybean.npz', 'run2__dropout-0.50__corn.npz', 'run5__dropout-0.1__soybean.npz', 'run7__deeper2__soybean.npz', 'run8__deeper3__soybean.npz', 'run9__deeper4__soybean.npz', 'run9__linear__soybean.npz']
#filename = sys.argv[1]
out = sys.argv[1]
convert_to_csv(filenames, out)


