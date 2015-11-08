lab1: lab2.o populate.o convergence.o
	g++ -O3 -o lab2 lab2.o populate.o convergence.o -fopenmp

lab1.o: lab2.cc lab2.h
	g++ -O3 -c lab2.cc

populate.o: populate.cc lab2.h
	g++ -O3 -c populate.cc

convergence.o: convergence.cc lab2.h
	g++ -O3 -c convergence.cc -fopenmp

clean:
	rm -rf *.o lab2
