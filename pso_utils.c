/* Utility functions for PSO */
#define _XOPEN_SOURCE 500 /* For definition of PI */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "pso.h"

/* Return a random number uniformly distributed between [min, max] */
float uniform(float min, float max)
{
    float normalized; 
    normalized = (float)rand()/(float)RAND_MAX;
    return (min + normalized * (max - min));
}

/* Use rand_r() to random between [0,1] with different seed for different threads */
float uniform_omp(float min, float max, unsigned int *seed)
{
    float normalized; 
    normalized = (float)rand_r(seed)/(float)RAND_MAX; /* rand_r is used for thread safety */
    return (min + normalized * (max - min));
}

/* Evaluate the eggholder function:
 *      f(x, y) = -(y + 47) * sin(sqrt(abs(y + x/2 + 47))) - x * sin(sqrt(abs(x - (y + 47))))
 *      Evaluated over the domain [-512, 512]
 *
 *      The function has a large number of local minima. 
 *
 *      The global minimum is f(x, y) = -959.6407 at (512, 404.2319)
 */ 
float pso_eval_eggholder(particle_t *particle)
{
    return -(particle->x[1] + 47) * sin(sqrt(fabsf(particle->x[0]/2 + particle->x[1]  + 47)))\
            - particle->x[0] * sin(sqrt(fabsf(particle->x[0] - (particle->x[1]  + 47))));
}

/* Evaluate the Schwefel function:
 *  f(x) = 418.9829 * d - \sum_{i = 1}^d x_i sin(sqrt(abs(x_i)))
 *  Evaluated over domain: x_i \in [-500, 500] for all i = 1, ..., d
 *
 *  d: number of dimensions
 *
 *  Function has many local minima. 
 *
 *  The global minimum is f(x) = 0 at (420.9687, ..., 420.9687)
 */
float pso_eval_schwefel(particle_t *particle)
{
    int i;
    float sum = 0;
    
    for (i = 0; i < particle->dim; i++)
        sum += particle->x[i] * sin(sqrt(fabsf(particle->x[i])));

    return 418.9829 * particle->dim - sum;
}

/* Evaluate the Holder table function: 
 *      f(x, y) = -|sinx * cosy * exp(|1 - (x^2 + y^2)^0.5/pi)|)|
 *      Evaluation domain: [-10, 10] 
 *
 *      The function is multi-modal and has four identical local minima:
 *          f(8.05502, 9.66459) = -19.2085,
 *          f(-8.05505, 9.66459) = -19.2085,
 *          f(8.05502, -9.66459) = -19.2085,
 *          f(-8.05505, -9.66459) = -19.2085,
 */
float pso_eval_holder_table(particle_t *particle)
{
    return -fabsf(sin(particle->x[0]) * cos(particle->x[1]) * exp(fabsf(1 - sqrt(pow(particle->x[0], 2) + pow(particle->x[1], 2))/M_PI)));
}
    
/* Evaluate the Rastrigin function:
 *      f(x) = A * d + \sum_i^d (x_i ^ 2 - A * cos(2* \pi * x_i))  
 *      A = 10
 *      x_i >= -5.12 and x_i <= 5.12, for all i in 1, 2, ..., d
 *      d: number of dimensions
 *      Global minimum of f(x) = 0 at x = 0
 */
float pso_eval_rastrigin(particle_t *particle)
{   
    int i;
    float fitness;
    
    fitness = 10 * particle->dim;
    for (i = 0; i < particle->dim; i++)
        fitness += pow(particle->x[i], 2) - 10 * cos(2 * M_PI * particle->x[i]);  
    
    return fitness;
}

/* Evaluate the Booth function:
 *      f(x, y) = (x + 2y - 7)^2 + (2x + y - 5)^2  
 *      Evaluation domain: [-10, 10] 
 *      Global minimum of f(1, 3) = 0
*/
float pso_eval_booth(particle_t *particle)
{
    return pow((particle->x[0] + 2 * particle->x[1] - 7), 2) 
           + pow((2 * particle->x[0] + particle->x[1] - 5), 2);
}

/* Evaluate particle's fitness using provided function. Return 0 on success, -1 otherwise */ 
int pso_eval_fitness(char *function, particle_t *particle, float *fitness)
{
    if (strcmp(function, "booth") == 0) {
        *fitness = pso_eval_booth(particle);
        return 0;
    }

    if (strcmp(function, "rastrigin") == 0) {
        *fitness = pso_eval_rastrigin(particle);
        return 0;
    }

    if (strcmp(function, "holder_table") == 0) {
        *fitness = pso_eval_holder_table(particle);
        return 0;
    }

    if (strcmp(function, "eggholder") == 0) {
        *fitness = pso_eval_eggholder(particle);
        return 0;
    }
    
    if (strcmp(function, "schwefel") == 0) {
        *fitness = pso_eval_schwefel(particle);
        return 0;
    }

    return -1;
}

/* Return index of best performing particle */
int pso_get_best_fitness(swarm_t *swarm)
{
    int i, g;
    float best_fitness = INFINITY;
    particle_t *particle;

    g = -1;
    for (i = 0; i < swarm->num_particles; i++) {
        particle = &swarm->particle[i];
        if (particle->fitness < best_fitness) {
            best_fitness = particle->fitness;
            g = i;
        }
    }
    return g;
}

/* Get best g of the swarm after an iteration  */
int pso_get_best_fitness_omp(swarm_t *swarm, int num_threads)
{
    int local_g[num_threads]; /* Each thread will store its local best g */
    int global_g = -1;
    float local_fitness[num_threads]; /* Each thread will store its local best fitness */
    // Init local fitness of each thread to infinity
    for (int i = 0; i < num_threads; i++)
        local_fitness[i] = INFINITY;
    float fitness;
    particle_t *particle;

#pragma omp parallel num_threads(num_threads) private(particle)
{
    int tid = omp_get_thread_num();
#pragma omp for
    for (int i = 0; i < swarm->num_particles; i++) {
        particle = &swarm->particle[i];
        if (particle->fitness < local_fitness[tid]) {
            local_fitness[tid] = particle->fitness;
            local_g[tid] = i;
        }
    }
}

    /* Find the best g of the swarm */
    // fitness = local_fitness[0]; 
    fitness = min(local_fitness, num_threads);
    global_g = local_g[0]; 
    for (int i = 0; i < num_threads; i++)
    {
        if (local_fitness[i] == fitness)
        {
            // fitness = local_fitness[i];
            global_g = local_g[i];
        }
    }
    
    return global_g;
}

/* Find min of the input array with known length */
float min(float * input, int length){
    float res = input[0]; 
    for (int i =0; i<length;i++){
        if (res > input[i]){
            res = input[i];
        }
    }
    return res;
}

/* Free swarm data structure */
void pso_free(swarm_t *swarm)
{
    int i;
    particle_t *particle;
    for (i = 0; i < swarm->num_particles; i++) { /* Free particle structures */
        particle = &(swarm->particle[i]);
        free((void *)particle->x);
        free((void *)particle->v);
        free((void *)particle->pbest);
    }
    
    free((void *)swarm);
    return;
}

/* Print current state of swarm */
void pso_print_particle(particle_t *particle)
{
    int j; 

    fprintf(stderr, "position: ");
    for (j = 0; j < particle->dim; j++)
        fprintf(stderr, "%.2f ", particle->x[j]);

    fprintf(stderr, "\nvelocity: ");
    for (j = 0; j < particle->dim; j++)
        fprintf(stderr, "%.2f ", particle->v[j]);

    fprintf(stderr, "\npbest: ");
    for (j = 0; j < particle->dim; j++)
        fprintf(stderr, "%.2f ", particle->pbest[j]);

    fprintf(stderr, "\nfitness: %.4f", particle->fitness);
    fprintf(stderr, "\ng: %d\n", particle->g);

    return;
}

void pso_print_swarm(swarm_t *swarm)
{
    int i;
    particle_t *particle;
    
    for (i = 0; i < swarm->num_particles; i++) {
         fprintf(stderr, "\nParticle: %d\n", i);
         particle = &swarm->particle[i];
         pso_print_particle(particle);
    }
     
    return;
}

/* Initialize PSO */
swarm_t *pso_init(char *function, int dim, int swarm_size, 
                  float xmin, float xmax)
{
    int i, j, g;
    int status;
    float fitness;
    swarm_t *swarm;
    particle_t *particle;

    swarm = (swarm_t *)malloc(sizeof(swarm_t));
    swarm->num_particles = swarm_size;
    swarm->particle = (particle_t *)malloc(swarm_size * sizeof(particle_t));
    if (swarm->particle == NULL)
        return NULL;

    for (i = 0; i < swarm->num_particles; i++) {
        particle = &swarm->particle[i];
        particle->dim = dim; 
        /* Generate random particle position */
        particle->x = (float *)malloc(dim * sizeof(float));
        for (j = 0; j < dim; j++)
           particle->x[j] = uniform(xmin, xmax);

       /* Generate random particle velocity */ 
        particle->v = (float *)malloc(dim * sizeof(float));
        for (j = 0; j < dim; j++)
            particle->v[j] = uniform(-fabsf(xmax - xmin), fabsf(xmax - xmin));

        /* Initialize best position for particle */
        particle->pbest = (float *)malloc(dim * sizeof(float));
        for (j = 0; j < dim; j++)
            particle->pbest[j] = particle->x[j];

        /* Initialize particle fitness */
        status = pso_eval_fitness(function, particle, &fitness);
        if (status < 0) {
            fprintf(stderr, "Could not evaluate fitness. Unknown function provided.\n");
            return NULL;
        }
        particle->fitness = fitness;

        /* Initialize index of best performing particle */
        particle->g = -1;
    }

    /* Get index of particle with best fitness */
    g = pso_get_best_fitness(swarm);
    for (i = 0; i < swarm->num_particles; i++) {
        particle = &swarm->particle[i];
        particle->g = g;
    }

    return swarm;
}

/* PSO init with parallel omp */
swarm_t *pso_init_omp(char *function, int dim, int swarm_size, 
                  float xmin, float xmax, int num_threads)
{
    int g;
    int status;
    float fitness;
    unsigned int seed = time(NULL);
    swarm_t *swarm;
    particle_t *particle;

    swarm = (swarm_t *)malloc(sizeof(swarm_t));
    swarm->num_particles = swarm_size;
    swarm->particle = (particle_t *)malloc(swarm_size * sizeof(particle_t));
    if (swarm->particle == NULL) {
        fprintf(stderr, "Malloc error\n");
        return NULL;
    }
// Start parallel section
#pragma omp parallel num_threads(num_threads) private(particle, status, fitness)
{
    int i, j;
    #pragma omp for
    for (i = 0; i < swarm->num_particles; i++) {
        seed += i; /* Get different seed for each thread*/
        particle = &swarm->particle[i];
        particle->dim = dim; 
        /* Generate random particle position */
        particle->x = (float *)malloc(dim * sizeof(float));
        for (j = 0; j < dim; j++)
           particle->x[j] = uniform_omp(xmin, xmax, &seed);   // Pass different seed to uniform_omp()

       /* Generate random particle velocity */ 
        particle->v = (float *)malloc(dim * sizeof(float));
        for (j = 0; j < dim; j++)
            particle->v[j] = uniform_omp(-fabsf(xmax - xmin), fabsf(xmax - xmin), &seed);

        /* Initialize best position for particle */
        particle->pbest = (float *)malloc(dim * sizeof(float));
        for (j = 0; j < dim; j++)
            particle->pbest[j] = particle->x[j];

        /* Initialize particle fitness */
        status = pso_eval_fitness(function, particle, &fitness);
        if (status < 0) {
            fprintf(stderr, "Could not evaluate fitness. Unknown function provided.\n");
            exit (EXIT_FAILURE);
        }
        particle->fitness = fitness;

        /* Initialize index of best performing particle */
        particle->g = -1;
    }
    

    /* One thread does extra work to get index of particle with best fitness */
#pragma omp single
    g = pso_get_best_fitness_omp(swarm, num_threads);
#pragma omp for /* Parallel loop. Independent particles */
    for (i = 0; i < swarm->num_particles; i++) {
        particle = &swarm->particle[i];
        particle->g = g;
    }
}

    return swarm;
}

