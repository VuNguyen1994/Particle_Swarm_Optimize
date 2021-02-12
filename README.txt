* Name: Dinh Nguyen, Tri Pham, Phi Manh Cuong
* Particle Swarm Optimizer PSO
*************************************
To run:
- Unzip the folder, and copy it to xunil
- At the directory, run make and it will compile the codes into executables
- Run pso by ./pso <function> <D> <swarm_size> <Xmin> <Xmax> <Max_iterations> <Num_threads>
For examples: ./pso schwefel 20 10000 -500 500 10000 16 will run schwefel function with Dimension D = 20,
swarm_size = 10000 particles, Xmin = -500, Xmax = 500, Max_iterations = 10000 and 16 threads.

It will generate 16 threads to divide the pso in parallel using OpenMP API.

**************************************