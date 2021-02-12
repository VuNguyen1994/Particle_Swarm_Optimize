/* Implementation of PSO using OpenMP.
 *
 * Author: Naga Kandasamy
 * Date: February 5, 2021
 *
 */
 #define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "pso.h"

int optimize_using_omp(char *function, int dim, int swarm_size, 
                       float xmin, float xmax, int max_iter, int num_threads)
{
    /* Initialize PSO */
    swarm_t *swarm;
    swarm = pso_init_omp(function, dim, swarm_size, xmin, xmax, num_threads);
    if (swarm == NULL) {
        fprintf(stderr, "Unable to initialize PSO\n");
        exit(EXIT_FAILURE);
    }

    int iter;
    float w, c1, c2;
    float curr_fitness;
    float r1, r2;
    unsigned int seed = time(NULL);
    particle_t *particle, *gbest;

    w = 0.79;
    c1 = 1.49;
    c2 = 1.49;
    iter = 0;
    int g=-1;

    while (iter < max_iter) {
    #pragma omp parallel num_threads(num_threads) shared (g) private(particle, gbest, r1, r2, curr_fitness, seed)
    {
        int i, j;
        #pragma omp for
        for (i = 0; i < swarm->num_particles; i++) {
            seed += i;  //Get different seed for each threads
            particle = &swarm->particle[i];
            gbest = &swarm->particle[particle->g];  /* Best performing particle from last iteration */ 
            for (j = 0; j < particle->dim; j++) {   /* Update this particle's state */
                r1 = (float)rand_r(&seed)/(float)RAND_MAX;    // Use rand_r instead for different seed
                r2 = (float)rand_r(&seed)/(float)RAND_MAX;
                /* Update particle velocity */
                particle->v[j] = w * particle->v[j]\
                                + c1 * r1 * (particle->pbest[j] - particle->x[j])\
                                + c2 * r2 * (gbest->x[j] - particle->x[j]);
                /* Clamp velocity */
                if ((particle->v[j] < -fabsf(xmax - xmin)) || (particle->v[j] > fabsf(xmax - xmin))) 
                    particle->v[j] = uniform_omp(-fabsf(xmax - xmin), fabsf(xmax - xmin), &seed); // Call uniform_omp with customized seed

                /* Update particle position */
                particle->x[j] = particle->x[j] + particle->v[j];
                if (particle->x[j] > xmax)
                    particle->x[j] = xmax;
                if (particle->x[j] < xmin)
                    particle->x[j] = xmin;
            } /* State update */
            
            /* Evaluate current fitness */
            pso_eval_fitness(function, particle, &curr_fitness);

            /* Update pbest */
            if (curr_fitness < particle->fitness) {
                particle->fitness = curr_fitness;
                for (j = 0; j < particle->dim; j++)
                    particle->pbest[j] = particle->x[j];
            }
        } /* Particle loop */
        /* Identify best performing particle */
    #pragma omp single
        g = pso_get_best_fitness_omp(swarm, num_threads);
    #pragma omp for
        for (i = 0; i < swarm->num_particles; i++) {
            particle = &swarm->particle[i];
            particle->g = g;
        }
    }

#ifdef SIMPLE_DEBUG
        /* Print best performing particle */
        fprintf(stderr, "\nIteration %d:\n", iter);
        pso_print_particle(&swarm->particle[g]);
#endif
        iter++;
    } /* End of iteration */

    /* Solve PSO */
    if (g >= 0) {
        fprintf(stderr, "Solution:\n");
        pso_print_particle(&swarm->particle[g]);
    }

    pso_free(swarm);
    return g;
}