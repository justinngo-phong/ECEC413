/* Reference implementation of PSO.
 *
 * Author: Justin Ngo, Harrison Muller 
 * Date modified: May 9, 2023
 *
 */

#define _POSIX_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "pso.h"

/* Solve PSO */
int pso_solve_omp(char *function, swarm_t *swarm, float xmax, float xmin, int max_iter, int num_threads)
{
	int i, j, iter, g;
	float w, c1, c2;
	float r1, r2;
	float curr_fitness;
	particle_t *particle, *gbest;

	w = 0.79;
	c1 = 1.49;
	c2 = 1.49;
	iter = 0;
	g = -1;
	unsigned int seed = time(NULL); // Seed the random number generator 
	omp_set_num_threads(num_threads);

	while (iter < max_iter) {
#pragma omp parallel private(i, j, particle, gbest, r1, r2, curr_fitness) firstprivate(w, c1, c2, xmax, xmin, g, seed) shared(swarm)
		{
			//seed = time(NULL) + omp_get_thread_num();
			//printf("%d, %d\n", iter, g);
#pragma omp for
			for (i = 0; i < swarm->num_particles; i++) {
				particle = &swarm->particle[i];
				gbest = &swarm->particle[particle->g];  /* Best performing particle from last iteration */ 
				for (j = 0; j < particle->dim; j++) {   /* Update this particle's state */
					r1 = (float)rand_r(&seed)/(float)RAND_MAX;
					r2 = (float)rand_r(&seed)/(float)RAND_MAX;
					/* Update particle velocity */
					particle->v[j] = w * particle->v[j]\
									 + c1 * r1 * (particle->pbest[j] - particle->x[j])\
									 + c2 * r2 * (gbest->x[j] - particle->x[j]);
					/* Clamp velocity */
					if ((particle->v[j] < -fabsf(xmax - xmin)) || (particle->v[j] > fabsf(xmax - xmin))) 
						particle->v[j] = uniform(-fabsf(xmax - xmin), fabsf(xmax - xmin));

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

#pragma omp critical
					{
					for (j = 0; j < particle->dim; j++)
						particle->pbest[j] = particle->x[j];
					}
				}
			} /* Particle loop */
#pragma omp barrier
		} /* End of parallel region */


		/* Identify best performing particle */
		g = pso_get_best_fitness_omp(swarm, num_threads);
		//g = pso_get_best_fitness(swarm);

#pragma omp parallel for private(i, particle) shared(swarm, g)
		for (i = 0; i < swarm->num_particles; i++) {
			particle = &swarm->particle[i];
			particle->g = g;
		}

#ifdef SIMPLE_DEBUG
		/* Print best performing particle */
		fprintf(stderr, "\nIteration %d:\n", iter);
		pso_print_particle(&swarm->particle[g]);
#endif

#pragma omp barrier
		iter++;
	} /* End of iteration */
	return g;
}


int optimize_using_omp(char *function, int dim, int swarm_size, 
		float xmin, float xmax, int max_iter, int num_threads)
{
	/* Initialize PSO */
	swarm_t *swarm;
	srand(time(NULL));
	swarm = pso_init_omp(function, dim, swarm_size, xmin, xmax, num_threads);
	//swarm = pso_init(function, dim, swarm_size, xmin, xmax);
	if (swarm == NULL) {
		fprintf(stderr, "Unable to initialize PSO\n");
		exit(EXIT_FAILURE);
	}

#ifdef VERBOSE_DEBUG
	pso_print_swarm(swarm);
#endif

	/* Solve PSO */
	int g; 
	g = pso_solve_omp(function, swarm, xmax, xmin, max_iter, num_threads);
	if (g >= 0) {
		fprintf(stderr, "Solution:\n");
		pso_print_particle(&swarm->particle[g]);
	}

	pso_free(swarm);
	return g;
}
