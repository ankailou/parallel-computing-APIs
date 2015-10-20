lab1: lab1.o populate.o convergence.o
	g++ -g -O3 -o lab1 lab1.o populate.o convergence.o -lpthread

lab1.o: lab1.cpp lab1.h
	g++ -g -O3 -c lab1.cpp

populate.o: populate.cpp lab1.h
	g++ -g -O3 -c populate.cpp

convergence.o: convergence.cpp lab1.h
	g++ -g -O3 -c convergence.cpp -lpthread

clean:
	rm -rf *.o lab1
