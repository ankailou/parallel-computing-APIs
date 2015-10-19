lab1: lab1.o populate.o convergence.o
	g++ -o lab1 lab1.o populate.o convergence.o

lab1.o: lab1.cpp lab1.h
	g++ -c lab1.cpp

populate.o: populate.cpp lab1.h
	g++ -c populate.cpp

convergence.o: convergence.cpp lab1.h
	g++ -c convergence.cpp

clean:
	rm -rf *.o lab1
