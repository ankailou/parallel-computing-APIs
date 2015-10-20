lab1: lab1.o populate.o convergence.o
	g++ -g -O3 -o lab1 lab1.o populate.o convergence.o -lpthread

lab1.o: lab1.cc lab1.h
	g++ -g -O3 -c lab1.cc

populate.o: populate.cc lab1.h
	g++ -g -O3 -c populate.cc

convergence.o: convergence.cc lab1.h
	g++ -g -O3 -c convergence.cc -lpthread

clean:
	rm -rf *.o lab1
