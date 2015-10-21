lab1: lab1.o populate.o convergence.o
	g++ -O3 -o lab1 lab1.o populate.o convergence.o

lab1.o: lab1.cc lab1.h
	g++ -O3 -c lab1.cc

populate.o: populate.cc lab1.h
	g++ -O3 -c populate.cc

convergence.o: convergence.cc lab1.h
	g++ -O3 -c convergence.cc

clean:
	rm -rf *.o lab1
