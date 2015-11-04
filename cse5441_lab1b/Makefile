lab1: convergence.o populate.o lab1.o
	g++ -O3 lab1.o populate.o convergence.o -lpthread -o lab1

convergence.o: convergence.cc lab1.h
	g++ -O3 convergence.cc -c -lpthread

populate.o: populate.cc lab1.h
	g++ -O3 populate.cc -c

lab1.o: lab1.cc lab1.h
	g++ -O3 lab1.cc -c

clean:
	rm -rf *.o lab1 *.gch
